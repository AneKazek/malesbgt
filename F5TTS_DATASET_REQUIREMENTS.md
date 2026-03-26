# F5-TTS Dataset Preparation & Training Data Requirements

## Quick Reference Summary

| Property | Specification |
|----------|--------------|
| **Sample Rate** | 24 kHz (target), any input rate supported via resampling |
| **Audio Format** | WAV, MP3, FLAC, M4A, OGG, Opus (any torchaudio-supported format) |
| **Audio Duration** | 0.3s to 30s per clip (filtered in __getitem__) |
| **Channels** | Any (auto-converted to mono) |
| **Text Format** | Pinyin (default), char, or custom vocabulary |
| **Dataset Format** | Arrow format (.arrow) with accompanying duration.json and vocab.txt |

---

## 1. Audio Specifications

### Sample Rate & Resampling
- **Target Sample Rate**: `24000 Hz` (hardcoded in configs)
- **Input Handling**: Any sample rate supported; automatic resampling during data loading
- **Resampling Method**: `torchaudio.transforms.Resample()`
- **Mono Conversion**: Multi-channel audio automatically averaged to mono

```python
# From src/f5_tts/model/dataset.py (lines 159-169)
if source_sample_rate != self.target_sample_rate:
    resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
    audio = resampler(audio)
```

### Audio Duration Requirements
- **Minimum**: 0.3 seconds
- **Maximum**: 30 seconds
- **Filtering**: Clips outside this range are automatically skipped during training
- **Duration Calculation**: Stored in `duration.json` during preprocessing

```python
# From src/f5_tts/model/dataset.py (lines 152-156)
if 0.3 <= duration <= 30:
    break  # valid sample
```

### Supported Audio Formats
From [preprocess.py](data/dataset_training/preprocess.py):
```python
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
```

---

## 2. Mel Spectrogram Configuration

The model converts audio to mel spectrograms with these fixed parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `target_sample_rate` | 24000 | Hardcoded across all configs |
| `n_mel_channels` | 100 | Frequency bins |
| `hop_length` | 256 | Samples per frame |
| `win_length` | 1024 | FFT window size |
| `n_fft` | 1024 | FFT size |
| `mel_spec_type` | vocos or bigvgan | Vocoder type |

**Frame Calculation**: `frames = duration * target_sample_rate / hop_length`
- A 1-second clip → `24000 / 256 ≈ 94` frames

---

## 3. Text Format & Tokenization

### Pinyin Tokenization (Default)
- **Used in**: Emilia_ZH_EN, datasetku, datasetku_smoketest datasets
- **Vocab File**: `vocab.txt` (one token per line)
- **Processing**: `convert_char_to_pinyin()` from `f5_tts.model.utils`
- **Example**: Chinese text `"你好"` → pinyin tokens

### Character Tokenization
- **Alternative Option**: `--pretrain` flag in prepare scripts
- **Use Case**: English/multilingual datasets (LibriTTS, Emilia_EN)
- **Vocab Generation**: Automatically extracted from training data

### Supported Tokenizers in Configs
```yaml
model:
  tokenizer: pinyin  # or 'char' for character-level
  tokenizer_path: null  # custom path if needed (should be vocab.txt)
```

---

## 4. Training Data Manifest Format (TSV)

### Format Specification
**File Type**: Tab-separated values (TSV)
**Required Columns**: 4 columns (tab-delimited)
**Header**: Present in file

### Column Structure
```
audio_path | text | dataset_source | speaker_id
```

**Example** (from data/dataset_training/combined/train_manifest_kaggle.tsv):
```tsv
data_indsp_news_tts/speech_extracted/SPK00_0000/SPK00_F_0000.wav	Reaksi berbeda disampaikan penasihat hukum pengungsi TimTim pro-integrasi Suhardi Somomoeljono.	indsp	spk00
data_indsp_news_tts/speech_extracted/SPK00_0000/SPK00_F_0001.wav	Konflik eksekutif-legislatif sebenarnya adalah hal yang lumrah dalam kehidupan berdemokrasi.	indsp	spk00
```

### Path Requirements
- **Audio Paths**: Must be **absolute paths** or resolvable relative to data directory
- **Path Validation**: Script attempts to resolve paths across different machines

---

## 5. Custom Dataset Preparation (CSV Format)

For creating custom datasets, use the CSV preparation script:

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /output/dataset/path [--pretrain]
```

### CSV Input Format
```csv
audio_file|text
/absolute/path/to/wavs/audio_0001.wav|Yo! Hello? Hello?
/absolute/path/to/wavs/audio_0002.wav|Hi, how are you doing today?
```

**Requirements**:
- **Header**: Required (literal string `audio_file|text`)
- **Delimiter**: Pipe character `|`
- **Paths**: Must be absolute paths
- **Text**: Arbitrary transcription text

### Output Dataset Format
After processing, the script generates:
```
output/
├── raw.arrow          # Processed dataset in Arrow format
├── duration.json      # Duration list for batch sampling
└── vocab.txt          # Vocabulary/tokenizer
```

---

## 6. Arrow Dataset Format (.arrow)

### Structure
The prepared dataset uses Apache Arrow format for efficient data loading.

### Associated Files
1. **raw.arrow** - Binary dataset file
   - Contains: List of records with `audio_path`, `text`, `duration`
   - Optional fields: `accent_id`, `lang_id`, `domain_id`

2. **duration.json** - Duration metadata
   ```json
   {
     "duration": [1.23, 2.45, 0.89, ...]
   }
   ```

3. **vocab.txt** - Tokenizer vocabulary
   ```
   [UNK]
   ！
   "
   #
   $
   ...
   ```

### Loading
```python
from datasets import Dataset
train_dataset = Dataset.from_file("data/dataset_name/raw.arrow")
with open("data/dataset_name/duration.json") as f:
    durations = json.load(f)["duration"]
```

---

## 7. Data Preprocessing Pipeline

### Step 1: Audio Loading & Validation
```python
# From prepare_csv_wavs.py
def process_audio_file(audio_path, text, polyphone):
    if not Path(audio_path).exists():
        return None
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return None
    return (audio_path, text, duration)
```

### Step 2: Text Preprocessing
For multi-lingual datasets:
- **Chinese Text**: Converted to pinyin
- **English Text**: Kept as-is (character tokenization)
- **Filtering**: Bad cases with repetitive characters removed

```python
# From prepare_emilia_v2.py
if repetition_found(text, length=4):
    continue  # Skip repetitive text
```

### Step 3: Batch Processing
- **Parallel Processing**: Uses `ProcessPoolExecutor` with 32 workers (configurable)
- **Batch Conversion**: Text converted to tokens in batches of 100
- **Format**: Auto-converts to vocoder-compatible format (vocos/bigvgan)

### Step 4: Dynamic Batch Sampling
During training:
- **Frame-based Batching**: Batches created to maintain ~38,400-307,200 frames per GPU
- **Max Samples Per Batch**: 64 sequences (configurable)
- **Sorting**: Samples sorted by duration for padding efficiency
- **Random Shuffling**: Deterministic shuffling with seed for reproducibility

```python
# From dataset.py - frame length calculation
frame_len = duration * target_sample_rate / hop_length
```

---

## 8. Configuration Examples

### Base Model Config (F5TTS_Base.yaml)
```yaml
datasets:
  name: Emilia_ZH_EN
  batch_size_per_gpu: 38400  # 8 GPUs → 307,200 total
  batch_size_type: frame     # frame-wise batching
  max_samples: 64            # max sequences per batch
  num_workers: 16

model:
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: vocos
```

### Small Model Config (F5TTS_Small.yaml)
```yaml
datasets:
  name: Emilia_ZH_EN
  batch_size_per_gpu: 38400
  max_samples: 32            # Smaller batches for smaller model
  
model:
  arch:
    dim: 768                 # vs 1024 for Base
    depth: 18                # vs 22 for Base
```

---

## 9. Data Augmentation & Quality Control

### Automatic Filtering
1. **Duration Filter**: 0.3s - 30s only
2. **Bad Case Detection**: Repetition patterns (e.g., "aaaa")
3. **Language-specific Filters**: Removes non-target characters

### Quality Checks in Preprocessing
- Audio file existence validation
- Duration extraction via `soundfile.info()` with ffprobe fallback
- Text encoding UTF-8 compatibility
- Frame count validation (min/max samples per batch)

### Resampling during Data Loading
- **On-the-fly**: Data resampled at load time (not during preprocessing)
- **Method**: `torchaudio.transforms.Resample()`
- **Reason**: Preserves original audio for potential re-use

---

## 10. Dataset Loading & Training Integration

### Load Dataset Function
```python
from f5_tts.model.dataset import load_dataset

train_dataset = load_dataset(
    dataset_name="Emilia_ZH_EN",
    tokenizer="pinyin",
    dataset_type="CustomDataset",
    mel_spec_kwargs={
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
    }
)
```

### Dataset Directory Structure
```
data/
├── Emilia_ZH_EN_pinyin/
│   ├── raw.arrow
│   ├── duration.json
│   └── vocab.txt
├── datasetku_pinyin/
│   ├── raw.arrow
│   ├── duration.json
│   └── vocab.txt
└── [other datasets]_[tokenizer]/
```

---

## 11. Batch Sampling Strategy

### Frame-wise Batching (Default)
- **Total Frames per Batch**: 38,400 frames per GPU (configurable)
- **Multi-GPU**: `batch_size_per_gpu * num_gpus = total_frames`
- **Example**: 8 GPUs × 38,400 = 307,200 total frames per batch
- **Adaptive**: Number of sequences varies based on duration

### Sample-wise Batching (Alternative)
- **Sequences per Batch**: Fixed number of clips
- **Padding**: More padding due to variable duration
- **Memory**: Less efficient than frame-wise

### Selection in Config
```yaml
datasets:
  batch_size_type: frame    # or 'sample'
```

---

## 12. Key Dataset Recipes

### For Chinese Text-to-Speech
1. **Use Dataset**: `datasetku_pinyin` or `Emilia_ZH_EN_pinyin`
2. **Tokenizer**: `"pinyin"`
3. **Duration**: 0.3-30 seconds
4. **Format**: Tab-separated: `audio_path | text`

### For English Text-to-Speech
1. **Use Dataset**: Dataset prepared with `--pretrain` flag
2. **Tokenizer**: `"char"`
3. **Duration**: Same 0.3-30 seconds
4. **Format**: Same tab-separated

### For Custom Multilingual Data
1. **Prepare**: Use prepare_csv_wavs.py with CSV
2. **Text Format**: Mix of pinyin and characters
3. **Custom Vocab**: Provide custom vocab.txt path
4. **Config**: Set `tokenizer_path: /path/to/vocab.txt`

---

## 13. Command Reference

### Prepare Existing Datasets
```bash
# Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# LibriTTS
python src/f5_tts/train/datasets/prepare_libritts.py

# LJSpeech
python src/f5_tts/train/datasets/prepare_ljspeech.py

# WeNetSpeech4TTS
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py
```

### Prepare Custom CSV Dataset
```bash
# Standard (finetune)
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /output/path

# Pretraining (character-level tokenizer)
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /output/path --pretrain

# Custom worker count
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /output/path --workers 16
```

### Start Training
```bash
accelerate config  # Setup multi-GPU config first

accelerate launch src/f5_tts/train/train.py --config-name F5TTS_Base.yaml
```

---

## 14. Important Notes

1. **Sample Rate is Hardcoded**: Always 24 kHz for mel-spectrogram generation
2. **Duration Filtering**: Done at data loading time, not preprocessing
3. **Memory Efficient**: Frame-based batching crucial for variable-duration sequences
4. **Deterministic**: Seeding at 666 for reproducible shuffling
5. **Arrow Format**: Chosen for streaming efficiency with parallel data loading
6. **Vocab Auto-generation**: Created during preprocessing for tokenization consistency
7. **Optional IDs**: `accent_id`, `lang_id`, `domain_id` supported for conditional training

---

## 15. Troubleshooting

| Issue | Solution |
|-------|----------|
| "sample rate hardcoded in server" | Always ensure input audio is resampled to 24kHz during preprocessing or use online resampling |
| OOM errors | Reduce `batch_size_per_gpu` or `max_samples` in config |
| Slow data loading | Check if duration.json provided; enable `num_workers` > 0 |
| Bad audio quality | Verify original sample rate ≥ 44.1kHz; check for corruption |
| Missing vocab.txt | Re-run prepare script; custom tokenizer needs explicit path |

---

## Source Code References

Key files referenced:
- [src/f5_tts/model/dataset.py](src/f5_tts/model/dataset.py) - Dataset classes and loading
- [src/f5_tts/train/datasets/prepare_csv_wavs.py](src/f5_tts/train/datasets/prepare_csv_wavs.py) - CSV dataset preparation
- [src/f5_tts/configs/F5TTS_Base.yaml](src/f5_tts/configs/F5TTS_Base.yaml) - Default configuration
- [data/dataset_training/preprocess.py](data/dataset_training/preprocess.py) - Data preprocessing utilities
- [src/f5_tts/train/README.md](src/f5_tts/train/README.md) - Training documentation
