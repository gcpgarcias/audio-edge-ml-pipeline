# CLAUDE.md — Audio Edge ML Pipeline

This file is read automatically by Claude Code at the start of every session.
Keep it concise (< 200 lines); link to topic files in `.claude/` for details.

---

## Project Overview

A **multi-stage ML pipeline** for deploying event-classification models on embedded
devices (Arduino Nicla Vision). Primary dataset: **fsc22**
(27 classes, 75 WAV files per class). The pipeline is capable of ingesting and preprocessing audio, image, text, tabular (numeric) and video datasets.

### Pipeline Stages

| # | Stage | Language / Tool | Status |
|---|-------|----------------|--------|
| 1 | Data ingestion | Python / FastAPI (`src/ingestion/api.py`) | Done |
| 2 | Data transformation | Python (`src/preprocessing/`) | **Done** |
| 3 | Model training | sklearn + Keras (`src/training/`) | **Done** |
| 4 | Model fine-tuning | sklearn GridSearchCV + Optuna Random Search (`src/training/tune.py`) | **Done** |
| 5 | Resource optimization | ONNX + onnxruntime (`src/optimization/`) | **Done** |
| 6 | Model compilation | Apache TVM | TODO |
| 7 | Model deployment | PlatformIO bare-C codegen (`src/deployment/`) | **In Progress** |
| 8 | Model monitoring | Streamlit (`src/monitoring/dashboard.py`) | TODO |
| 9 | Model updating | — | TODO |

| # | Helper           | Language / Tool                          | Status   |
|---|------------------|------------------------------------------|----------|
| 1 | Model evaluation | MLflow local file store (`mlruns/`)      | **Done** |
| 2 | Model selection  | MLflow + CLI (`src/training/select.py`)  | **Done** |

---

## Stage 7 — Deployment

**Chosen path**: bare-C PlatformIO codegen (Arduino mbed framework on Nicla Vision).
OpenMV+TFLite was evaluated and deferred. TVM/X-CUBE-AI/TFLM also deferred.

### Key files

```
src/deployment/
├── deploy.py                     # CLI entry point — routes .onnx → OnnxToC
└── codegen/
    ├── onnx_to_c.py              # ONNX fp32 → C inference + PlatformIO project
    └── model_to_c.py             # Shared templates: features.h/c, audio.cpp, main.cpp

deploy/nicla_cnn/                 # Generated project (do not hand-edit; re-run deploy.py)
```

### Deploy CLI

```bash
python -m src.deployment.deploy \
  --model   data/models/optimized/<experiment>/<run>/model_fp32.onnx \
  --board   nicla_vision \
  --max-ram 460 \
  --output  deploy/nicla_cnn \
  --features-dir data/processed/fsc22_melspec_val_2

cd deploy/nicla_cnn && pio run --target upload && \
  until [ -e /dev/cu.usbmodem11201 ]; do sleep 0.5; done && \
  pio device monitor
```

### Critical architecture constraints (Nicla Vision / STM32H747)

- **Use fp32 ONNX** — static int8 has same arena size but more codegen complexity.
- **AXI SRAM**: 512 KB at `0x24000000`. mbed Arduino takes ~330 KB BSS, leaving ~180 KB.
- **Arena budget**: `2 × max_activation` must fit in ~180 KB.
  - `first_stride=2`: Conv1 out = (16,20,251) = 314 KB → **does not fit**.
  - `first_stride=4`: Conv1 out = (16,10,126) = 79 KB → arena ~157 KB → **fits** ✓
- **Custom linker script** (`nicla_vision.ld`) generated automatically — keeps stack at
  top of physical 512 KB (do not set LENGTH > 512K or stack lands in non-existent RAM).
- **PDM gain**: `PDM.setGain(32)` in `audio_init()`. Default gain is near-zero.
- **Feature extraction must match librosa exactly**:
  - `librosa.power_to_db(mel, ref=np.max)` + `_normalize()` → values in `[0, 1]`
  - `center=True` (librosa default): `n_frames = 1 + n_samples // hop`; frame window
    starts at `fi*hop - n_fft/2` with zero-padding outside `[0, n_samples)`.
  - Duration: **5.0 s exactly** → 80 000 samples → 501 frames at hop=160.
- **Conv padding**: ONNX `pads` = `[h_top, w_left, h_bottom, w_right]` (asymmetric).
  The C primitive `ml_conv2d_nchw_linear` accepts all 4 values separately.

### Optimization CLI (Stage 5)

```bash
python -m src.optimization.optimize \
  --model-path  data/models/<run>/model.keras \
  --model-name  cnn \
  --features    data/processed/fsc22_melspec_augmented_train_2 \
  --features-eval data/processed/fsc22_melspec_val_2 \
  --class-filter Fire Thunderstorm Silence Speaking BirdChirping \
  --experiment  fsc22-nicla-5-classes-optimization \
  --output-dir  data/models/optimized
```

---

## Stage 2 Architecture — Feature Extraction

All feature extraction lives in `src/preprocessing/`. Design: **modality ×
feature-type** grid with a registry/factory pattern.

### Key Abstractions (`src/preprocessing/feature_extraction/base.py`)

- **`FeatureSet`** — uniform output container; `labels` and `cluster_assignments`
  are both `Optional[np.ndarray]` (supervised / unsupervised / semi-supervised).
- **`BaseFeatureExtractor`** — ABC with `extract(Optional[Path], **kwargs)`
  and `extract_dataset(loader, max_samples) → FeatureSet`.
- **`BaseDatasetLoader`** — ABC yielding `(Optional[Path], Optional[str], dict)`.

### Registered Extractors (18 total)

```
audio_classical   audio_mel_spec    audio_waveform  audio_cqt  audio_mfcc_seq
image_classical   image_pixels      image_mobilenet_v2
text_tfidf        text_bow          text_char_ngram text_sentence_embed text_bert_tokens
tabular_classical tabular_polynomial
video_classical   video_frame_seq   video_mobilenet_v2_seq
```

---

## Stage 3 — Training, Evaluation, Selection

### Registered Trainers

```
Classical:  svm  lda  decision_tree  random_forest  knn  kmeans  pca_svm  pca_lda  pca_knn
Deep:       mlp  cnn  rnn  transformer
```

### CNNTrainer architecture notes

- `first_stride`: stride of block 0 Conv2D. Use **4** for Nicla Vision deployment.
- `second_stride`: stride of block 1 Conv2D. Use **2** to further reduce activations.
- `filters`: list or int per block. `[16,16,16]` fits on Nicla.

### Stage 3 CLI

```bash
python -m src.training.train --config config/training.yaml
mlflow ui --backend-store-uri mlruns/
```

### Shortlist files

```text
data/models/shortlist_<experiment>.json           pre-opt (from train.py)
data/models/tuned/shortlist_<experiment>.json     post-tune (from tune.py)
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| librosa | Audio loading, MFCCs, spectrograms, CQT |
| scikit-learn | ColumnTransformer, scalers, PolynomialFeatures |
| tensorflow | MobileNetV2, model training |
| tensorflow-metal | Apple Silicon GPU backend |
| onnx / onnxruntime | Stage 5 optimization + Stage 7 codegen source |
| mlflow | Experiment tracking |
| PyYAML | YAML config parsing |

---

## Environment Notes

- Activate the venv before running anything.
- Apple Silicon: `tensorflow-metal` is in `requirements.txt`.
- Do **not** commit `.npy` / `.json` processed data — large and reproducible.
- Serial port for Nicla Vision: `/dev/cu.usbmodem11201` (may increment after each flash).
