from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text, transcribe


def _load_cfg(path: Path):
    cfg = OmegaConf.load(str(path))
    model_cls = getattr(__import__("f5_tts.model", fromlist=[cfg.model.backbone]), cfg.model.backbone)
    model_cfg = OmegaConf.to_container(cfg.model.arch, resolve=True)
    mel_type = str(cfg.model.mel_spec.mel_spec_type)
    tokenizer = str(cfg.model.tokenizer)
    if tokenizer == "custom":
        vocab_file = cfg.model.tokenizer_path
    else:
        vocab_file = ""
    return model_cls, model_cfg, mel_type, vocab_file


def _flatten_prompt_sets(prompt_json: dict) -> list[tuple[str, str]]:
    items = []
    for group in ["short", "medium", "long", "tricky"]:
        prompts = prompt_json.get(group, [])
        for idx, prompt in enumerate(prompts, start=1):
            items.append((f"{group}_{idx}", str(prompt)))
    return items


def _repetition_ratio(text: str) -> float:
    toks = [t for t in text.lower().replace("\n", " ").split(" ") if t]
    if not toks:
        return 1.0
    counts = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    return max(counts.values()) / len(toks)


def _score_output(
    wave: np.ndarray | None,
    sample_rate: int,
    prompt: str,
    *,
    use_asr: bool,
    tmp_wav_path: Path,
) -> dict:
    if wave is None or wave.size == 0:
        return {
            "is_empty": True,
            "duration_sec": 0.0,
            "rms": 0.0,
            "text": "",
            "too_short_transcript": None,
            "high_repetition": None,
            "possible_skip": None,
        }

    duration_sec = float(len(wave) / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(wave))) + 1e-12)
    is_empty = duration_sec < 0.25 or rms < 1e-4

    transcript = ""
    too_short_transcript = None
    high_repetition = None
    possible_skip = None

    if use_asr and not is_empty:
        wav_tensor = torch.from_numpy(wave).to(dtype=torch.float32).unsqueeze(0)
        torchaudio.save(str(tmp_wav_path), wav_tensor, sample_rate)
        transcript = transcribe(str(tmp_wav_path))
        ratio = (len(transcript.strip()) + 1) / (len(prompt.strip()) + 1)
        too_short_transcript = len(transcript.strip()) < max(8, int(0.25 * len(prompt.strip())))
        high_repetition = _repetition_ratio(transcript) > 0.45
        possible_skip = ratio < 0.45

    return {
        "is_empty": is_empty,
        "duration_sec": duration_sec,
        "rms": rms,
        "text": transcript,
        "too_short_transcript": too_short_transcript,
        "high_repetition": high_repetition,
        "possible_skip": possible_skip,
    }


def main():
    parser = argparse.ArgumentParser(description="Mini A/B sanity eval for baseline vs hybrid checkpoints.")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--hybrid-config", required=True)
    parser.add_argument("--hybrid-ckpt", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--prompts", default="src/f5_tts/eval/hybrid_ab_prompts.json")
    parser.add_argument("--out-dir", default="ckpts/hybrid_eval_sanity")
    parser.add_argument("--nfe-step", type=int, default=16)
    parser.add_argument("--cfg-strength", type=float, default=1.75)
    parser.add_argument("--use-asr", action="store_true")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.prompts).open("r", encoding="utf-8") as f:
        prompt_json = json.load(f)
    prompts = _flatten_prompt_sets(prompt_json)

    baseline_cls, baseline_cfg, baseline_mel_type, baseline_vocab = _load_cfg(Path(args.baseline_config))
    hybrid_cls, hybrid_cfg, hybrid_mel_type, hybrid_vocab = _load_cfg(Path(args.hybrid_config))

    if baseline_mel_type != hybrid_mel_type:
        raise ValueError("baseline and hybrid mel_spec_type must match for A/B sanity comparison")

    vocoder = load_vocoder(vocoder_name=baseline_mel_type, device=args.device)

    baseline_model = load_model(
        baseline_cls,
        baseline_cfg,
        args.baseline_ckpt,
        mel_spec_type=baseline_mel_type,
        vocab_file=baseline_vocab,
        device=args.device,
    )
    hybrid_model = load_model(
        hybrid_cls,
        hybrid_cfg,
        args.hybrid_ckpt,
        mel_spec_type=hybrid_mel_type,
        vocab_file=hybrid_vocab,
        device=args.device,
    )

    ref_audio, ref_text = preprocess_ref_audio_text(args.ref_audio, args.ref_text)

    report = {"prompts": [], "aggregate": {"baseline": {}, "hybrid": {}}}

    for prompt_id, prompt in prompts:
        row = {"id": prompt_id, "prompt": prompt}
        for tag, model in (("baseline", baseline_model), ("hybrid", hybrid_model)):
            wave, sr, _ = infer_process(
                ref_audio,
                ref_text,
                prompt,
                model,
                vocoder,
                mel_spec_type=baseline_mel_type,
                nfe_step=args.nfe_step,
                cfg_strength=args.cfg_strength,
                speed=1.0,
                device=args.device,
            )

            out_wav = out_dir / f"{prompt_id}_{tag}.wav"
            tmp_asr_wav = out_dir / f"{prompt_id}_{tag}_asr_tmp.wav"

            if wave is not None:
                wav_tensor = torch.from_numpy(wave).to(dtype=torch.float32).unsqueeze(0)
                torchaudio.save(str(out_wav), wav_tensor, sr)

            metrics = _score_output(
                wave,
                sr,
                prompt,
                use_asr=args.use_asr,
                tmp_wav_path=tmp_asr_wav,
            )
            metrics["wav_path"] = str(out_wav)
            row[tag] = metrics

        report["prompts"].append(row)

    for tag in ("baseline", "hybrid"):
        total = len(report["prompts"])
        empty_count = sum(1 for item in report["prompts"] if item[tag]["is_empty"])
        asr_short = sum(
            1
            for item in report["prompts"]
            if item[tag]["too_short_transcript"] is True
        )
        asr_repeat = sum(
            1
            for item in report["prompts"]
            if item[tag]["high_repetition"] is True
        )
        asr_skip = sum(
            1
            for item in report["prompts"]
            if item[tag]["possible_skip"] is True
        )
        report["aggregate"][tag] = {
            "total_prompts": total,
            "empty_generation_count": empty_count,
            "too_short_transcript_count": asr_short,
            "high_repetition_count": asr_repeat,
            "possible_skip_count": asr_skip,
            "asr_enabled": bool(args.use_asr),
        }

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved sanity report to: {report_path}")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()