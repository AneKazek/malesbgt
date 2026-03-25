from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from f5_tts.model import CFM
from f5_tts.model.checkpoint_audit import format_checkpoint_audit, summarize_checkpoint_load
from f5_tts.model.utils import get_tokenizer


def _load_state_dict_from_checkpoint(ckpt_path: Path) -> dict[str, torch.Tensor]:
    suffix = ckpt_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        ckpt = {"ema_model_state_dict": load_file(str(ckpt_path), device="cpu")}
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    if "model_state_dict" in ckpt:
        model_sd = ckpt["model_state_dict"]
    else:
        ema_sd = ckpt.get("ema_model_state_dict", {})
        model_sd = {k.replace("ema_model.", ""): v for k, v in ema_sd.items() if k not in {"initted", "step", "update"}}

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        model_sd.pop(key, None)

    return model_sd


def _build_cfm_from_config(config_path: Path) -> CFM:
    cfg = OmegaConf.load(str(config_path))
    model_cls = getattr(__import__("f5_tts.model", fromlist=[cfg.model.backbone]), cfg.model.backbone)

    tokenizer = cfg.model.tokenizer
    if tokenizer == "custom":
        tokenizer_path = cfg.model.tokenizer_path
    else:
        tokenizer_path = cfg.datasets.name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    model = CFM(
        transformer=model_cls(
            **OmegaConf.to_container(cfg.model.arch, resolve=True),
            text_num_embeds=vocab_size,
            mel_dim=cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=OmegaConf.to_container(cfg.model.mel_spec, resolve=True),
        vocab_char_map=vocab_char_map,
        **OmegaConf.to_container(cfg.model.get("cfm_experiment", {}), resolve=True),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Audit baseline->hybrid checkpoint compatibility.")
    parser.add_argument("--hybrid-config", required=True, help="Path to hybrid yaml config.")
    parser.add_argument("--checkpoint", required=True, help="Path to baseline checkpoint (.pt/.safetensors).")
    parser.add_argument("--json-out", default="", help="Optional path to save full audit json.")
    args = parser.parse_args()

    hybrid_config = Path(args.hybrid_config)
    ckpt_path = Path(args.checkpoint)

    model = _build_cfm_from_config(hybrid_config)
    source_sd = _load_state_dict_from_checkpoint(ckpt_path)

    summary = summarize_checkpoint_load(model, source_sd)
    print(format_checkpoint_audit(summary, max_group_lines=30))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved json summary to: {out_path}")


if __name__ == "__main__":
    main()