from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file as load_safetensors
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from f5_tts.model import CFM, DiT, HybridDiT
from f5_tts.model.backbones.hybrid_dit import copy_shared_weights, load_partial_state_dict_safely
from f5_tts.model.dataset import collate_fn, load_dataset
from f5_tts.model.utils import get_tokenizer


def parse_int_list(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill warmup: freeze everything except new Mamba mixers and train distillation first."
    )

    parser.add_argument("--dataset-name", type=str, default="datasetku")
    parser.add_argument("--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "byte", "custom"])
    parser.add_argument("--tokenizer-path", type=str, default=None)

    parser.add_argument("--teacher-ckpt", type=str, default="", help="Optional checkpoint (.pt/.safetensors) to load teacher")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--conv-layers", type=int, default=0)

    parser.add_argument("--use-bidi", action="store_true")
    parser.add_argument("--mamba-layers", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)

    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--n-mel-channels", type=int, default=100)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--mel-spec-type", type=str, default="vocos", choices=["vocos", "bigvgan"])

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means run full epochs")

    parser.add_argument("--lambda-distill-out", type=float, default=1.0)
    parser.add_argument("--lambda-distill-hidden", type=float, default=0.0)
    parser.add_argument("--distill-hidden-layers", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default="ckpts/distill_freeze_runs")
    parser.add_argument("--run-name", type=str, default="")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_run_dir(base_dir: str, run_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() if run_name.strip() else f"distill_{stamp}"
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


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


def maybe_to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    return x


def _load_raw_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        obj = load_safetensors(path, device="cpu")
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(obj, dict):
        if "ema_model_state_dict" in obj and isinstance(obj["ema_model_state_dict"], dict):
            return obj["ema_model_state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj).__name__}")
    return obj


def _extract_backbone_state(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    backbone_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        k = key
        for prefix in ("ema_model.", "model.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]

        if k.startswith("transformer."):
            k = k[len("transformer."):]

        backbone_state[k] = value
    return backbone_state


def load_teacher_checkpoint(teacher: DiT, ckpt_path: str) -> int:
    raw_state = _load_raw_state_dict(ckpt_path)
    backbone_state = _extract_backbone_state(raw_state)
    n_loaded = load_partial_state_dict_safely(teacher, backbone_state)
    return n_loaded


def freeze_all_then_unfreeze_new_mixers(model: CFM, mamba_layers: list[int]) -> tuple[int, int]:
    for p in model.parameters():
        p.requires_grad = False

    transformer = model.transformer
    depth = len(transformer.transformer_blocks)

    for idx in mamba_layers:
        if 0 <= idx < depth:
            block = transformer.transformer_blocks[idx]
            mixer = getattr(block, "mixer", None)
            if mixer is not None:
                for p in mixer.parameters():
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_curves(
    output_png: str,
    total_loss: list[float],
    flow_loss: list[float],
    distill_loss: list[float],
) -> None:
    x = list(range(1, len(total_loss) + 1))
    smooth_window = max(1, min(50, len(total_loss) // 10 if len(total_loss) >= 10 else 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x, total_loss, label="loss_total", alpha=0.45)
    plt.plot(x, flow_loss, label="loss_flow", alpha=0.45)
    plt.plot(x, distill_loss, label="loss_distill_out", alpha=0.45)

    if smooth_window > 1:
        plt.plot(x, moving_average(total_loss, smooth_window), label=f"total moving avg ({smooth_window})", linewidth=2)

    plt.title("Distill Warmup Loss Curves")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)

    run_dir = build_run_dir(args.output_dir, args.run_name)
    cfg_path = os.path.join(run_dir, "run_config.txt")
    csv_path = os.path.join(run_dir, "loss_history.csv")
    plot_path = os.path.join(run_dir, "distill_loss.png")
    student_ckpt_path = os.path.join(run_dir, "student_distill_last.pt")

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

    mamba_layers = [x for x in parse_int_list(args.mamba_layers) if 0 <= x < args.depth]
    distill_hidden_layers = parse_int_list(args.distill_hidden_layers)

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

    teacher_transformer = DiT(**arch_common).to(device)
    student_transformer = HybridDiT(
        **arch_common,
        use_mamba=True,
        mamba_layers=mamba_layers,
        use_bidi=args.use_bidi,
        inject_sinpos=True,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        capture_hidden_for_distill=(args.lambda_distill_hidden > 0 and len(distill_hidden_layers) > 0),
    ).to(device)

    loaded_teacher = 0
    if args.teacher_ckpt.strip():
        loaded_teacher = load_teacher_checkpoint(teacher_transformer, args.teacher_ckpt.strip())
        print(f"Loaded teacher checkpoint tensors: {loaded_teacher}")

    copied_student = copy_shared_weights(student_transformer, teacher_transformer)
    print(f"Copied teacher->student shared tensors: {copied_student}")

    model = CFM(
        transformer=student_transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
        use_distill=True,
        lambda_distill_out=args.lambda_distill_out,
        lambda_distill_hidden=args.lambda_distill_hidden,
        distill_hidden_layers=distill_hidden_layers,
        teacher_transformer=teacher_transformer,
    ).to(device)

    trainable, total = freeze_all_then_unfreeze_new_mixers(model, mamba_layers)
    print(f"Trainable params after freeze policy: {trainable}/{total}")
    if trainable == 0:
        raise RuntimeError("No trainable parameters left. Check mamba_layers/use_bidi and architecture depth.")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=args.lr)

    global_step = 0
    hist_total: list[float] = []
    hist_flow: list[float] = []
    hist_distill_out: list[float] = []

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
            total_loss, _, _ = model(
                mel_spec,
                text=text_inputs,
                lens=mel_lengths,
                accent_id=accent_id,
                lang_id=lang_id,
                domain_id=domain_id,
            )
            total_loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm_(optim_params, args.max_grad_norm)
            optimizer.step()

            global_step += 1

            detail = model.last_loss_dict
            loss_total = float(detail.get("loss_total", total_loss.detach()).item())
            loss_flow = float(detail.get("loss_flow", torch.tensor(0.0, device=device)).item())
            loss_distill = float(detail.get("loss_distill_out", torch.tensor(0.0, device=device)).item())

            hist_total.append(loss_total)
            hist_flow.append(loss_flow)
            hist_distill_out.append(loss_distill)

            pbar.set_postfix(step=global_step, loss=f"{loss_total:.6f}", distill=f"{loss_distill:.6f}")

            if args.max_updates > 0 and global_step >= args.max_updates:
                print(f"Reached max updates: {args.max_updates}")
                break

        if args.max_updates > 0 and global_step >= args.max_updates:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_total", "loss_flow", "loss_distill_out"])
        for i in range(len(hist_total)):
            writer.writerow([i + 1, hist_total[i], hist_flow[i], hist_distill_out[i]])

    save_curves(plot_path, hist_total, hist_flow, hist_distill_out)

    torch.save(
        {
            "student_transformer_state_dict": model.transformer.state_dict(),
            "config": vars(args),
            "steps": global_step,
        },
        student_ckpt_path,
    )

    if hist_total:
        print(f"Training finished. Steps: {len(hist_total)}")
        print(f"Final total loss: {hist_total[-1]:.6f}")
        print(f"Best total loss:  {min(hist_total):.6f}")
        print(f"Final distill loss: {hist_distill_out[-1]:.6f}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved plot:  {plot_path}")
    print(f"Saved ckpt:  {student_ckpt_path}")


if __name__ == "__main__":
    main()
