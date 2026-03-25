#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 tidak ditemukan di PATH"
  exit 1
fi

python3.11 -m venv .venv_new
source .venv_new/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Base deps pinned for Python 3.11 + torch 2.6/cu124
pip install -r requirements-py311-torch260-cu124.txt

# Install project without re-resolving dependencies
pip install -e . --no-deps

# Requested install order
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

python - <<'PY'
import importlib
import torch
import torchaudio
print('python ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchaudio', torchaudio.__version__)
for mod in ['causal_conv1d', 'mamba_ssm', 'accelerate', 'hydra']:
    importlib.import_module(mod)
    print('import ok:', mod)
PY

echo
echo "Environment siap."
echo "Contoh smoke test training:"
echo "  source .venv_new/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=fp16 .venv_new/bin/python src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name datasetku --finetune --tokenizer pinyin --batch_size_per_gpu 256 --batch_size_type frame --max_samples 1 --grad_accumulation_steps 4 --learning_rate 1e-5 --epochs 1 --num_warmup_updates 20 --save_per_updates 20 --last_per_updates 10 --keep_last_n_checkpoints 2 --bnb_optimizer"
