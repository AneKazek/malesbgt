from __future__ import annotations

import importlib.util

import torch
from torch import nn

from f5_tts.model import CFM, DiT, HybridDiT


class DummyMelSpec(nn.Module):
    def __init__(self, n_mel_channels: int):
        super().__init__()
        self.n_mel_channels = n_mel_channels

    def forward(self, wav):
        raise RuntimeError("DummyMelSpec.forward should not be called in this smoke test")


def make_text(batch: int, max_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    text = torch.full((batch, max_len), -1, dtype=torch.long, device=device)
    lengths = torch.randint(low=max_len // 2, high=max_len, size=(batch,), device=device)
    for i, length in enumerate(lengths.tolist()):
        text[i, :length] = torch.randint(low=0, high=vocab_size, size=(length,), device=device)
    return text


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_mamba = importlib.util.find_spec("mamba_ssm") is not None
    use_mamba = bool(has_mamba and device.type == "cuda")

    depth = 10
    hidden_index = 7
    batch = 2
    seq_len = 24
    mel_dim = 32
    vocab_size = 64

    mel = torch.randn(batch, seq_len, mel_dim, device=device)
    lens = torch.tensor([seq_len, seq_len - 5], device=device, dtype=torch.long)
    text = make_text(batch, max_len=12, vocab_size=vocab_size, device=device)

    teacher_tf = DiT(
        dim=64,
        depth=depth,
        heads=4,
        dim_head=16,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=32,
        conv_layers=0,
        dropout=0.0,
        attn_mask_enabled=False,
    ).to(device)
    student_tf = HybridDiT(
        dim=64,
        depth=depth,
        heads=4,
        dim_head=16,
        ff_mult=2,
        mel_dim=mel_dim,
        text_num_embeds=vocab_size,
        text_dim=32,
        conv_layers=0,
        dropout=0.0,
        attn_mask_enabled=False,
        use_mamba=use_mamba,
        mamba_layers=[0, 1, 2, 3] if use_mamba else [],
    ).to(device)

    cfm = CFM(
        transformer=student_tf,
        num_channels=mel_dim,
        mel_spec_module=DummyMelSpec(mel_dim),
        vocab_char_map={str(i): i for i in range(vocab_size)},
        use_distill=True,
        lambda_distill_out=0.0,
        lambda_distill_hidden=0.5,
        distill_hidden_layers=[hidden_index],
        teacher_transformer=teacher_tf,
    ).to(device)

    assert teacher_tf.capture_hidden_for_distill is True, "CFM did not enable teacher hidden capture"
    assert student_tf.capture_hidden_for_distill is True, "CFM did not enable student hidden capture"

    loss, _, _ = cfm(mel, text=text, lens=lens)
    hidden_loss = float(cfm.last_loss_dict["loss_distill_hidden"])

    teacher_hidden = teacher_tf.last_hidden_states
    student_hidden = student_tf.last_hidden_states

    assert teacher_hidden is not None, "Teacher DiT did not expose last_hidden_states"
    assert student_hidden is not None, "Student backbone did not expose last_hidden_states"
    assert len(teacher_hidden) == depth, f"Expected {depth} teacher hidden states, got {len(teacher_hidden)}"
    assert len(student_hidden) == depth, f"Expected {depth} student hidden states, got {len(student_hidden)}"
    assert 0 <= hidden_index < len(teacher_hidden), f"Teacher hidden index {hidden_index} is invalid"
    assert 0 <= hidden_index < len(student_hidden), f"Student hidden index {hidden_index} is invalid"
    assert (
        teacher_hidden[hidden_index].shape == student_hidden[hidden_index].shape
    ), "Teacher and student hidden shapes do not match at the requested layer"
    assert hidden_loss > 0.0, "loss_distill_hidden stayed at 0.0"

    teacher_tf.set_capture_hidden_for_distill(False)
    with torch.no_grad():
        _ = teacher_tf(
            x=mel,
            cond=mel,
            text=text,
            time=torch.rand(batch, device=device),
            mask=torch.arange(seq_len, device=device).unsqueeze(0) < lens.unsqueeze(1),
        )
    assert teacher_tf.last_hidden_states is None, "Teacher DiT kept stale hidden states when capture was disabled"

    print(f"device={device.type} use_mamba={use_mamba} has_mamba_pkg={has_mamba}")
    print(f"teacher_hidden_len={len(teacher_hidden)}")
    print(f"student_hidden_len={len(student_hidden)}")
    print(f"teacher_hidden_{hidden_index}_shape={tuple(teacher_hidden[hidden_index].shape)}")
    print(f"student_hidden_{hidden_index}_shape={tuple(student_hidden[hidden_index].shape)}")
    print(f"loss_total={float(loss.detach()):.6f}")
    print(f"loss_distill_hidden={hidden_loss:.6f}")
    print("teacher_hidden_disabled=None")
    print("smoke_teacher_dit_hidden_distill=PASS")


if __name__ == "__main__":
    main()
