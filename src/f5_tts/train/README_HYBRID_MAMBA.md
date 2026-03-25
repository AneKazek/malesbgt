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
