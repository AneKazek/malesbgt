# Hybrid Mamba Experiment (Safe Patch)

This patch adds a conservative experimental path without replacing baseline F5-TTS behavior.

## New Backbone

- Use `model.backbone: HybridDiT`
- Baseline-compatible mode:
  - `use_mamba: false`
  - `mamba_layers: []`
- Minimal hybrid mode:
  - `use_mamba: true`
  - `mamba_layers: [10, 11]` for `depth: 22`

## Optional CFM Experimental Losses

Under `model.cfm_experiment`:

- Distillation:
  - `use_distill`
  - `lambda_distill_out`
  - `lambda_distill_hidden`
  - `distill_hidden_layers`
- Auxiliary CTC:
  - `use_ctc`
  - `lambda_ctc`
  - `ctc_on_generated_only`
  - `ctc_layer_index`
- Accent adversarial:
  - `use_accent_adv`
  - `lambda_adv`
  - `adv_num_classes`
  - `adv_pooling`
  - `adv_feature_layer`

If optional labels (`accent_id`, `lang_id`, `domain_id`) are not present in the dataset,
adversarial loss stays zero and training continues.

## Safety Notes

- Public inference scripts are unchanged.
- `CFM.forward` contract remains `(loss, cond, pred)`.
- Additional loss details are exposed through `model.last_loss_dict`.
- Checkpoint loading keeps strict behavior first and falls back to non-strict only if strict fails.

## Smoke Test

Run:

```bash
python src/f5_tts/scripts/smoke_hybrid_mamba.py
```

The script checks baseline and hybrid single-step forward/backward paths,
plus optional distill/ctc/adv branches in a lightweight setup.

## Conservative Presets

New conservative configs for early safety runs:

- `F5TTS_SANITY_HYBRID_1K.yaml`
  - `use_mamba: true`, `mamba_layers: [10, 11]`
  - all auxiliary losses OFF (`distill/ctc/adv` and all lambdas `0`)
- `F5TTS_HYBRID_DISTILL_5K.yaml`
  - `use_mamba: true`, `mamba_layers: [10, 11]`
  - output distill ON (`use_distill: true`, small `lambda_distill_out`)
  - hidden distill / ctc / adv OFF
- `F5TTS_BASELINE_CONTROL_1K.yaml`
  - `use_mamba: false` and all auxiliary losses OFF

## Checkpoint Audit

Use the audit script to validate baseline -> hybrid non-strict loading:

```bash
python src/f5_tts/scripts/audit_hybrid_checkpoint.py \
  --hybrid-config src/f5_tts/configs/F5TTS_SANITY_HYBRID_1K.yaml \
  --checkpoint /path/to/baseline_checkpoint.pt \
  --json-out ckpts/hybrid_ckpt_audit.json
```

The report includes:

- total parameter coverage and load ratio
- expected vs suspicious missing keys
- expected vs suspicious unexpected keys
- grouped key summaries and copied/new-init module summaries

## Mini A/B Sanity Eval

Run lightweight A/B detection for obvious failures:

```bash
python src/f5_tts/eval/eval_hybrid_sanity.py \
  --baseline-config src/f5_tts/configs/F5TTS_BASELINE_CONTROL_1K.yaml \
  --baseline-ckpt /path/to/baseline_checkpoint.pt \
  --hybrid-config src/f5_tts/configs/F5TTS_SANITY_HYBRID_1K.yaml \
  --hybrid-ckpt /path/to/hybrid_checkpoint.pt \
  --ref-audio /path/to/ref.wav \
  --ref-text "" \
  --prompts src/f5_tts/eval/hybrid_ab_prompts.json \
  --out-dir ckpts/hybrid_eval_sanity \
  --use-asr
```

Checks include empty generation, repetition signals, and optional ASR-based short/skipped output flags.
