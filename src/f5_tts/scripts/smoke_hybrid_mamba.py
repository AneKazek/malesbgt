from __future__ import annotations

import importlib.util

import torch
from torch import nn

from f5_tts.model import CFM, DiT, HybridDiT
from f5_tts.model.backbones.hybrid_dit import load_partial_state_dict_safely


class DummyMelSpec(nn.Module):
    def __init__(self, n_mel_channels: int):
        super().__init__()
        self.n_mel_channels = n_mel_channels

    def forward(self, wav):
        raise RuntimeError("DummyMelSpec.forward should not be called in this smoke test")


def make_text(batch: int, max_len: int, vocab_size: int, device: torch.device):
    text = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
    lengths = torch.randint(low=max_len // 2, high=max_len, size=(batch,), device=device)
    for i, l in enumerate(lengths.tolist()):
        text[i, :l] = torch.randint(low=0, high=vocab_size, size=(l,), device=device)
    return text


def run_single_step(model: CFM, mel: torch.Tensor, text: torch.Tensor, lens: torch.Tensor, **kwargs):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, _, _ = model(mel, text=text, lens=lens, **kwargs)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_mamba = importlib.util.find_spec("mamba_ssm") is not None

    batch = 2
    seq = 96
    mel_dim = 32
    vocab_size = 64

    mel = torch.randn(batch, seq, mel_dim, device=device)
    lens = torch.tensor([seq, seq - 8], device=device, dtype=torch.long)
    text = make_text(batch, max_len=48, vocab_size=vocab_size, device=device)

    baseline_tf = DiT(
        dim=128,
        depth=8,
        heads=4,
        dim_head=32,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=64,
        conv_layers=0,
        attn_mask_enabled=False,
    ).to(device)
    baseline = CFM(transformer=baseline_tf, num_channels=mel_dim, mel_spec_module=DummyMelSpec(mel_dim)).to(device)
    baseline_loss = run_single_step(baseline, mel, text, lens)

    use_mamba = bool(has_mamba)
    hybrid_tf = HybridDiT(
        dim=128,
        depth=8,
        heads=4,
        dim_head=32,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=64,
        conv_layers=0,
        attn_mask_enabled=False,
        use_mamba=use_mamba,
        mamba_layers=[3, 4] if use_mamba else [],
        capture_hidden_for_distill=True,
    ).to(device)

    hybrid = CFM(transformer=hybrid_tf, num_channels=mel_dim, mel_spec_module=DummyMelSpec(mel_dim)).to(device)
    hybrid_loss = run_single_step(hybrid, mel, text, lens)

    teacher_tf = DiT(
        dim=128,
        depth=8,
        heads=4,
        dim_head=32,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=64,
        conv_layers=0,
        attn_mask_enabled=False,
    ).to(device)
    student_tf = HybridDiT(
        dim=128,
        depth=8,
        heads=4,
        dim_head=32,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=64,
        conv_layers=0,
        attn_mask_enabled=False,
        use_mamba=use_mamba,
        mamba_layers=[3, 4] if use_mamba else [],
        capture_hidden_for_distill=True,
    ).to(device)

    copied = load_partial_state_dict_safely(student_tf, teacher_tf.state_dict())
    distill = CFM(
        transformer=student_tf,
        num_channels=mel_dim,
        mel_spec_module=DummyMelSpec(mel_dim),
        use_distill=True,
        distill_hidden_layers=[3, 4],
        teacher_transformer=teacher_tf,
    ).to(device)
    distill_loss = run_single_step(distill, mel, text, lens)

    ctc = CFM(
        transformer=student_tf,
        num_channels=mel_dim,
        mel_spec_module=DummyMelSpec(mel_dim),
        vocab_char_map={str(i): i for i in range(vocab_size)},
        use_ctc=True,
    ).to(device)
    ctc_loss = run_single_step(ctc, mel, text, lens)

    adv = CFM(
        transformer=student_tf,
        num_channels=mel_dim,
        mel_spec_module=DummyMelSpec(mel_dim),
        use_accent_adv=True,
        adv_num_classes=6,
    ).to(device)
    adv_loss_no_label = run_single_step(adv, mel, text, lens)

    print(f"baseline_step_loss={baseline_loss:.6f}")
    print(f"hybrid_step_loss={hybrid_loss:.6f} use_mamba={use_mamba}")
    print(f"distill_step_loss={distill_loss:.6f} copied_params={copied}")
    print(f"ctc_step_loss={ctc_loss:.6f}")
    print(f"adv_missing_label_step_loss={adv_loss_no_label:.6f}")


if __name__ == "__main__":
    main()
