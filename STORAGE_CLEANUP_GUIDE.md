# Storage Cleanup Guide untuk Kaggle 2x T4

## Masalahnya
- `/kaggle/temp` has 10-15GB limits yang mudah penuh
- venv, dataset cache, checkpoint semua terakumulasi

## Solusi Sebelum Jalankan Training

### 1. Bersihkan Temp Paling Aggressive
```bash
# Jalankan ini cell sebelum training cell
import subprocess
import shutil
from pathlib import Path

# Kill any running Python processes using /kaggle/temp
subprocess.run(['pkill', '-f', '/kaggle/temp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Hapus temp tapi keep venv + repo
temp_path = Path('/kaggle/temp')
if temp_path.exists():
    for item in temp_path.glob('*'):
        if item.name not in ['.venv', 'kcv-tts']:
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
                print(f'✓ Deleted: {item.name}')
            except Exception as e:
                print(f'✗ Failed to delete {item.name}: {e}')

# Clear pip cache
subprocess.run(['/kaggle/temp/.venv/bin/pip', 'cache', 'purge'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('✓ Pip cache cleared')
```

### 2. Reduce Arrow Dataset Cache Footprint
```bash
# Add this to dataset prep cell - BEFORE conversion
import os
os.environ['HF_DATASETS_CACHE'] = '/kaggle/temp/hf_cache'
os.environ['HF_HOME'] = '/kaggle/temp/hf_cache'

# Disable HF telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
```

### 3. Use `/kaggle/working` ONLY for Checkpoints
Already done in your notebook - checkpoints go to:
```
/kaggle/working/ckpts/<MODEL>_vocos_pinyin_<DATASET>/
```
This survives notebook reruns, temp doesn't.

### 4. Monitoring During Training
```python
import os
import subprocess

def check_storage():
    result = subprocess.run(['df', '-h', '/kaggle/temp', '/kaggle/working'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
# Call during training to monitor:
# check_storage()
```

## Estimated Storage Usage per Epoch

| Component | Size |
|-----------|------|
| Dataset (5000 samples) | ~3-5GB |
| Single checkpoint | ~800MB-1GB |
| venv | ~2-3GB |
| HF cache (temp) | ~1-2GB |
| **Total needed** | **~7-12GB** |

Kaggle provides ~15GB per notebook run, so:
- **10 epochs**: Should be fine (~8-10GB total)
- **30 epochs**: Risky (might exceed limits)

## Recommended Workflow

1. **Before running training cell:**
   ```python
   # Clear temp (but keep venv + repo)
   # Clear pip cache
   # Check df -h to verify free space
   ```

2. **During training:**
   ```bash
   watch -n 60 'df -h /kaggle/temp /kaggle/working | tail -2'
   ```

3. **If storage gets dangerously full (<1GB free):**
   - Stop training
   - Delete old checkpoints manually
   - Or reduce `keep_last_n_checkpoints` even further (but value >= 1)

## Storage Optimization Flags

Your notebook already has these. Don't change:
- `ckpts.save_per_updates=500` ← Save frequency optimal
- `ckpts.keep_last_n_checkpoints=1` ← Minimal retention (already aggressive)
- `ckpts.log_samples=False` ← No audio generation I/O

Potential future reductions (if still crashing):
- `save_per_updates=800` (save less frequently)
- `ckpts.save_dir` → move to even smaller temp location
- Reduce `BATCH_SIZE_PER_GPU` (not recommended, hurts training)

## Next Run Checklist

- [ ] Run cleanup cell before training
- [ ] Verify `df -h /kaggle/temp` shows >5GB free
- [ ] Run training cell
- [ ] Monitor `df -h` if feels risky
- [ ] After training, copy best checkpoint to `/kaggle/working/final/` for safety
