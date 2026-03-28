from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from f5_tts.model import CFM, DiT, HybridDiT
from f5_tts.model.dataset import collate_fn, load_dataset
from f5_tts.model.utils import get_tokenizer


def parse_mamba_layers(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight F5-TTS training loop and save a training-loss plot."
    )

    parser.add_argument("--dataset-name", type=str, default="datasetku", help="Dataset name under data/<name>_pinyin")
    parser.add_argument("--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "byte", "custom"])
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path for custom tokenizer vocab.txt")

    parser.add_argument("--backbone", type=str, default="HybridDiT", choices=["HybridDiT", "DiT"])
    parser.add_argument("--use-mamba", action="store_true", help="Enable mamba layers for HybridDiT")
    parser.add_argument("--mamba-layers", type=str, default="0,1,2,3,4,5,6,7")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--conv-layers", type=int, default=0)

    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--use-bidi", action="store_true", help="Use bidirectional mamba mixer")

    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--n-mel-channels", type=int, default=100)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--mel-spec-type", type=str, default="vocos", choices=["vocos", "bigvgan"])

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7.5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means run full epochs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default="ckpts/train_loss_plots")
    parser.add_argument("--run-name", type=str, default="")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    return x


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    if window <= 1:
        return values[:]
    out = []
    cumsum = 0.0
    queue = []
    for v in values:
        queue.append(v)
        cumsum += v
        if len(queue) > window:
            cumsum -= queue.pop(0)
        out.append(cumsum / len(queue))
    return out


def build_run_dir(base_dir: str, run_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() if run_name.strip() else f"run_{stamp}"
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)

    run_dir = build_run_dir(args.output_dir, args.run_name)
    csv_path = os.path.join(run_dir, "loss_history.csv")
    plot_path = os.path.join(run_dir, "training_loss.png")
    cfg_path = os.path.join(run_dir, "run_config.txt")

    with open(cfg_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")

    tokenizer_source = args.dataset_name if args.tokenizer != "custom" else (args.tokenizer_path or "")
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_source, args.tokenizer)

    mel_spec_kwargs = {
        "target_sample_rate": args.target_sample_rate,
        "n_mel_channels": args.n_mel_channels,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "n_fft": args.n_fft,
        "mel_spec_type": args.mel_spec_type,
    }

    train_dataset = load_dataset(args.dataset_name, args.tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device.type == "cuda"),
        batch_size=args.batch_size,
        shuffle=True,
    )

    mamba_layers = parse_mamba_layers(args.mamba_layers)
    mamba_layers = [x for x in mamba_layers if 0 <= x < args.depth]

    arch_common = dict(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        ff_mult=args.ff_mult,
        mel_dim=args.n_mel_channels,
        text_num_embeds=vocab_size,
        text_dim=args.text_dim,
        conv_layers=args.conv_layers,
        attn_backend="torch",
        attn_mask_enabled=False,
        checkpoint_activations=False,
    )

    if args.backbone == "HybridDiT":
        transformer = HybridDiT(
            **arch_common,
            use_mamba=args.use_mamba,
            mamba_layers=mamba_layers,
            use_bidi=args.use_bidi,
            inject_sinpos=True,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
        )
    else:
        transformer = DiT(**arch_common)

    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    loss_values: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for batch in pbar:
            mel_spec = batch["mel"].permute(0, 2, 1).to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            text_inputs = batch["text"]
            accent_id = maybe_to_device(batch.get("accent_id"), device)
            lang_id = maybe_to_device(batch.get("lang_id"), device)
            domain_id = maybe_to_device(batch.get("domain_id"), device)

            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(
                mel_spec,
                text=text_inputs,
                lens=mel_lengths,
                accent_id=accent_id,
                lang_id=lang_id,
                domain_id=domain_id,
            )
            loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            global_step += 1
            loss_value = float(loss.detach().item())
            loss_values.append(loss_value)
            pbar.set_postfix(step=global_step, loss=f"{loss_value:.6f}")

            if args.max_updates > 0 and global_step >= args.max_updates:
                print(f"Reached max updates: {args.max_updates}")
                break

        if args.max_updates > 0 and global_step >= args.max_updates:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, v in enumerate(loss_values, start=1):
            writer.writerow([i, v])

    smooth_window = max(1, min(50, len(loss_values) // 10 if len(loss_values) >= 10 else 1))
    smooth = moving_average(loss_values, smooth_window)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, label="train loss", alpha=0.45)
    if smooth_window > 1:
        plt.plot(range(1, len(smooth) + 1), smooth, label=f"moving avg ({smooth_window})", linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    if loss_values:
        print(f"Training finished. Steps: {len(loss_values)}")
        print(f"Final loss: {loss_values[-1]:.6f}")
        print(f"Best loss:  {min(loss_values):.6f}")
    print(f"Saved loss CSV:  {csv_path}")
    print(f"Saved loss plot: {plot_path}")


if __name__ == "__main__":
    main()
