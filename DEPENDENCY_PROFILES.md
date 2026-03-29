# Dependency Profiles (Local vs Kaggle)

This repository has multiple dependency profiles to avoid local/Kaggle version conflicts.

## Recommended Profiles

- Torch 2.8 + cu128 + Python 3.11 (recommended for current notebook flow):
  - requirements-torch28-cu12-localmatch.txt
- Torch 2.6 + cu124 + Python 3.11:
  - requirements-py311-torch260-cu124.txt
- Torch 2.10 fallback profile:
  - requirements-kaggle-torch210.txt

## Local Setup (uv + existing venv)

1. Create venv with Python 3.11.
2. Install profile:

```bash
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-torch28-cu12-localmatch.txt
```

## Kaggle Setup

1. Use Python 3.11 venv in notebook.
2. Install the same profile for parity with local:

```bash
uv pip install --python /kaggle/temp/kcv-tts/.venv/bin/python --index-strategy unsafe-best-match -r /kaggle/temp/kcv-tts/requirements-torch28-cu12-localmatch.txt
```

## Notes

- requirements.txt is a broad pinned snapshot; avoid using it as the primary install source for Kaggle training runs.
- Editable install in requirements.txt is now portable (-e .) instead of a machine-specific absolute path.
- Keep torch/torchaudio/torchvision in one profile family (do not mix cu124 and cu128 stacks).
