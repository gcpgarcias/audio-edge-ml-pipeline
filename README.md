# Audio Edge ML Pipeline

A multi-stage ML pipeline for training and deploying audio event-classification models on embedded devices (Arduino Nicla Vision / STM32H747). Primary dataset: **fsc22** (27 classes, 75 WAV files per class).

---

## Pipeline Overview

| Stage | Description | Status |
| --- | --- | --- |
| 1 | Data ingestion (FastAPI) | Done |
| 2 | Feature extraction | Done |
| 3 | Model training | Done |
| 4 | Model fine-tuning | Done |
| 5 | Resource optimization | Done |
| 6 | Model compilation | TODO |
| 7 | Bare-C PlatformIO deployment | In Progress |
| 8 | Model monitoring (Streamlit) | TODO |
| 9 | Model updating | TODO |

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Apple Silicon: `tensorflow-metal` is included in `requirements.txt`.  
> Do **not** commit `.npy` / `.json` processed data — it is large and reproducible.

---

## Stage 1 — Data Ingestion

Start the FastAPI ingestion server:

```bash
uvicorn src.ingestion.api:app --reload --host 0.0.0.0 --port 8000
```

Accepts file uploads and exposes feature query endpoints.

---

## Stage 2 — Feature Extraction

```bash
# Single run
python -m src.preprocessing.pipeline \
  --loader <loader> \
  --extractor <extractor> \
  [--dataset <path>] \
  [--output <dir>] \
  [--split train|val|test] \
  [--classes <c1> <c2> ...] \
  [--max-samples <n>]

# Config-driven (recommended)
python -m src.preprocessing.pipeline --config config/feature_extraction.yaml
```

### Registered loaders

`birdeep`, `audio_folder`, `image_folder`, `text_folder`, `video_folder`, `tabular_csv`

### Registered extractors (18 total)

| Modality | Extractors |
| --- | --- |
| Audio | `audio_classical`, `audio_mel_spec`, `audio_waveform`, `audio_cqt`, `audio_mfcc_seq` |
| Image | `image_classical`, `image_pixels`, `image_mobilenet_v2` |
| Text | `text_tfidf`, `text_bow`, `text_char_ngram`, `text_sentence_embed`, `text_bert_tokens` |
| Tabular | `tabular_classical`, `tabular_polynomial` |
| Video | `video_classical`, `video_frame_seq`, `video_mobilenet_v2_seq` |

### Audio augmentation

```bash
python -m src.preprocessing.augment --config config/augmentation.yaml
```

Config keys: `loader`, `dataset`, `manifest`, `split`, `seed`, `output_dir`, `n_augments`, `preserve_length`, `level_match_db`, `augmentations`, `class_overrides`.

---

## Stage 3 — Model Training

```bash
# Single run
python -m src.training.train \
  --features <dir> \
  --model <name> \
  [--output <dir>] \
  [--val-split <float>] \
  [--experiment <name>] \
  [--run-name <name>] \
  [--max-samples <n>] \
  [--param KEY=VALUE ...]

# Config-driven (recommended)
python -m src.training.train --config config/training.yaml
```

### Registered trainers

| Type | Names |
| --- | --- |
| Classical | `svm`, `lda`, `decision_tree`, `random_forest`, `knn`, `kmeans`, `pca_svm`, `pca_lda`, `pca_knn` |
| Deep | `mlp`, `cnn`, `rnn`, `transformer` |

### View experiments

```bash
mlflow ui --backend-store-uri mlruns/
```

---

## Stage 4 — Hyperparameter Tuning

```bash
python -m src.training.tune --config config/tuning.yaml
```

Config keys: `output_dir`, `experiment`, `mlflow_uri`, `cv`, `scoring`, `n_trials`, `sweep_epochs`, `seed`, `pruner`, `shortlist`, and per-run `grid` (classical → GridSearchCV) or `search_space` (deep → Optuna random search).

---

## Stage 4b — Model Selection

```bash
# Pre-optimization shortlist
python -m src.training.select \
  --experiment <name> \
  --output <dir> \
  [--metric val_f1_macro] \
  [--top-n 5] \
  [--min-accuracy <float>]

# Post-optimization selection
python -m src.training.select \
  --post-opt \
  --shortlist data/models/tuned/shortlist_<experiment>.json \
  --opt-dir data/models/optimized \
  [--max-size-kb <float>]
```

Shortlist files:

- Pre-opt: `data/models/shortlist_<experiment>.json`
- Post-tune: `data/models/tuned/shortlist_<experiment>.json`

---

## Stage 5 — Resource Optimization

Benchmarks fp32, dynamic INT8, static INT8, and float16 ONNX modes. Selects the smallest model within an accuracy-drop threshold.

```bash
# From shortlist
python -m src.optimization.optimize \
  --shortlist data/models/tuned/shortlist_<experiment>.json \
  [--features-eval <dir>] \
  [--output-dir data/models/optimized] \
  [--experiment <name>] \
  [--max-accuracy-drop 0.02]

# Single model
python -m src.optimization.optimize \
  --model-path data/models/<run>/model.keras \
  --model-name <name> \
  --features <dir> \
  [--features-eval <dir>] \
  [--class-filter <c1> <c2> ...] \
  [--experiment <name>] \
  [--output-dir data/models/optimized]
```

Writes `optimization_report.json` per model to the output directory.

---

## Stage 7 — Deployment (Nicla Vision)

Generates a bare-C PlatformIO project from an ONNX model.

```bash
python -m src.deployment.deploy \
  --model data/models/optimized/<experiment>/<run>/model_fp32.onnx \
  --board nicla_vision \
  --output deploy/nicla_cnn \
  [--max-ram 460] \
  [--features-dir data/processed/fsc22_melspec_val_2] \
  [--class-filter <c1> <c2> ...] \
  [--duration 5.0] \
  [--sample-rate 16000] \
  [--n-mels 40] \
  [--n-fft 512] \
  [--hop-length 160]
```

Flash and monitor:

```bash
cd deploy/nicla_cnn && pio run --target upload && \
  until [ -e /dev/cu.usbmodem11201 ]; do sleep 0.5; done && \
  pio device monitor
```

### Architecture constraints (STM32H747)

- Use **fp32 ONNX** — static int8 has the same arena size but more codegen complexity.
- AXI SRAM: 512 KB at `0x24000000`. mbed Arduino takes ~330 KB BSS → ~180 KB available.
- `first_stride=4` required: Conv1 output = (16,10,126) = 79 KB → arena ~157 KB ✓
- Feature extraction must match librosa exactly (`center=True`, `power_to_db(ref=np.max)`, normalized to `[0,1]`).
- PDM gain: `PDM.setGain(12)` — default gain is near-zero.

---

## Stage 8 — Monitoring

```bash
streamlit run src/monitoring/dashboard.py
```

Real-time device monitoring dashboard with auto-refresh.

---

## Tools

### `tools/generate_split.py` — Stratified split manifest

```bash
python tools/generate_split.py \
  --input <dataset-dir> \
  [--loader audio_folder] \
  [--train 0.70] \
  [--val 0.15] \
  [--test 0.15] \
  [--seed 42] \
  [--force]
```

Outputs `split_manifest.json` in the dataset directory.

---

### `tools/record_dataset.py` — Record labelled audio via Nicla

```bash
python tools/record_dataset.py \
  --class <label> \
  [--n 30] \
  [--output data/raw/fsc22_device] \
  [--port <serial-port>] \
  [--baud 115200] \
  [--source-dir <dir>] \
  [--seed <int>]
```

Streams audio from the Nicla over serial and saves labelled `.wav` files.

---

### `tools/receive_wav.py` — Receive raw PCM from Nicla

```bash
python tools/receive_wav.py \
  [--port <serial-port>] \
  [--baud 115200] \
  [--out <path>] \
  [--experiment default] \
  [--count 1]
```

Receives raw PCM bytes over serial and saves as `.wav`.

---

### `tools/receive_mel.py` — Receive mel spectrogram from Nicla

```bash
python tools/receive_mel.py \
  [--features data/processed/fsc22_melspec_val_2] \
  [--port <serial-port>] \
  [--baud 115200] \
  [--experiment default] \
  [--label <class>] \
  [--save <path>] \
  [--load <path>]
```

Receives a mel spectrogram from the device and compares it against training samples for feature alignment verification.

---

### `tools/evaluate_device.py` — Evaluate flashed model on device

```bash
python tools/evaluate_device.py \
  --manifest <split-manifest-path> \
  [--source-dir <dir>] \
  [--loader audio_folder] \
  [--split test] \
  [--classes <c1> <c2> ...] \
  [--optimization-report <path>] \
  [--port <serial-port>] \
  [--baud 115200] \
  [--experiment fsc22-device-eval] \
  [--run-name <name>] \
  [--mlflow-uri mlruns/] \
  [--no-mlflow]
```

Plays back audio files, reads device predictions over serial, and logs a classification report to MLflow.

---

### `tools/gen_prototypes.py` — Generate C prototype arrays

```bash
python tools/gen_prototypes.py \
  --features <processed-features-dir> \
  --classes <c1> <c2> ... \
  --out-dir <dir>
```

Computes per-class mean mel spectrograms and writes them as `uint8` C arrays for on-device MSE diagnostics.

---

## Key File Locations

```text
config/
├── feature_extraction.yaml     # Stage 2 config
├── augmentation.yaml           # Stage 2 augmentation config
├── training.yaml               # Stage 3 config
├── tuning.yaml                 # Stage 4 config
└── experiments/                # Per-experiment overrides

data/
├── raw/                        # Source datasets
├── processed/                  # Extracted features (.npy)
└── models/
    ├── shortlist_<exp>.json    # Pre-opt shortlist
    ├── tuned/                  # Post-tune models + shortlists
    └── optimized/              # ONNX models + optimization_report.json

deploy/nicla_cnn/               # Generated PlatformIO project (do not hand-edit)
mlruns/                         # MLflow experiment store
```
