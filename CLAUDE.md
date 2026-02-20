# CLAUDE.md — Audio Edge ML Pipeline

This file is read automatically by Claude Code at the start of every session.
Keep it concise (< 200 lines); link to topic files in `.claude/` for details.

---

## Project Overview

A **10-stage ML pipeline** for deploying event-classification models on embedded
devices (Arduino Nicla Vision). Primary dataset: **BIRDeep_AudioAnnotations**
(36 bird species, WAV + PNG pairs, YOLOv8 bbox annotations, pre-split CSVs).

### Pipeline Stages

| # | Stage | Language / Tool | Status |
|---|-------|----------------|--------|
| 1 | Data ingestion | Python / FastAPI (`src/ingestion/api.py`) | Done |
| 2 | Data transformation | Python (`src/preprocessing/`) | **Done** |
| 3 | Model training | TensorFlow (`src/training/`) | Needs expansion |
| 4 | Model evaluation | MLflow (`docker/`) | Needs expansion |
| 5 | Model selection | scikit-learn | TODO |
| 6 | Model optimization | LiteRT (`src/optimization/quantize.py`) | Needs expansion |
| 7 | Model compilation | Apache TVM | TODO |
| 8 | Model deployment | PlatformIO (`src/deployment/edge_simulator.py`) | TODO |
| 9 | Model monitoring | Streamlit (`src/monitoring/dashboard.py`) | Needs review |
| 10 | Model updating | — | TODO |

---

## Stage 2 Architecture — Feature Extraction

All feature extraction lives in `src/preprocessing/`. Design: **modality ×
feature-type** grid with a registry/factory pattern.

### Key Abstractions (`src/preprocessing/feature_extraction/base.py`)

- **`FeatureSet`** — uniform output container; `labels` and `cluster_assignments`
  are both `Optional[np.ndarray]` (supervised / unsupervised / semi-supervised;
  use `-1` in labels for unlabelled samples).
- **`BaseFeatureExtractor`** — ABC with `extract(Optional[Path], **kwargs)`
  and `extract_dataset(loader, max_samples) → FeatureSet`.
- **`BaseDatasetLoader`** — ABC yielding `(Optional[Path], Optional[str], dict)`;
  `None` path signals in-memory data (tabular rows, JSON text).

### Registry (`src/preprocessing/feature_extraction/registry.py`)

```python
@register
class MyExtractor(BaseFeatureExtractor):
    name         = "my_extractor"   # globally unique
    feature_type = "classical"      # "classical" | "deep"
    modality     = "audio"          # "audio"|"image"|"text"|"tabular"|"video"
```

`get("my_extractor")` returns the class; `list_extractors()` lists all 18 names.

### Registered Extractors (18 total)

```
audio_classical   audio_mel_spec    audio_waveform  audio_cqt  audio_mfcc_seq
image_classical   image_pixels      image_mobilenet_v2
text_tfidf        text_bow          text_char_ngram text_sentence_embed text_bert_tokens
tabular_classical tabular_polynomial
video_classical   video_frame_seq   video_mobilenet_v2_seq
```

### Stateful Extractors (fit before transform)

Text classical (TF-IDF, BoW, char n-gram) and tabular extractors override
`extract_dataset()` for corpus-level fit. Calling `extract()` before
`extract_dataset()` raises `RuntimeError`.

---

## File Map

```
src/preprocessing/
├── pipeline.py                        # FeaturePipeline orchestrator + CLI (flag + --config)
├── config.py                          # PipelineConfig / ExperimentConfig + load_config()
├── audio_processor.py                 # Ingestion-layer shim → AudioMelSpectrogram
├── feature_extraction/
│   ├── base.py                        # FeatureSet, BaseFeatureExtractor, BaseDatasetLoader
│   ├── registry.py                    # @register, get(), list_extractors()
│   ├── __init__.py                    # Re-exports all concrete classes (triggers registration)
│   ├── audio/
│   │   ├── classical.py               # AudioClassicalExtractor — 302-dim flat vector
│   │   └── deep.py                    # AudioMelSpectrogram, AudioWaveform, AudioCQT, AudioMFCCSequence
│   ├── image/
│   │   ├── classical.py               # ImageClassicalExtractor — HOG + LBP + histogram + GLCM
│   │   └── deep.py                    # ImagePixels, ImageMobileNetV2 (lazy-loaded)
│   ├── text/
│   │   ├── classical.py               # TextTFIDFExtractor, TextBoWExtractor, TextCharNgramExtractor
│   │   └── deep.py                    # TextSentenceEmbedding (all-MiniLM-L6-v2), TextBERTTokens
│   ├── tabular/
│   │   ├── classical.py               # TabularClassicalExtractor — ColumnTransformer pipeline
│   │   └── deep.py                    # TabularPolynomialExtractor — degree-2 poly expansion
│   └── video/
│       ├── classical.py               # VideoClassicalExtractor — HOG+LBP+hist mean/std + opt. flow
│       └── deep.py                    # VideoFrameSequence (T,H,W,C), VideoMobileNetV2Sequence (T,1280)
└── dataset_loaders/
    ├── audio_folder_loader.py         # AudioFolderLoader — class-per-subfolder .wav/.flac/…
    ├── birdeep_loader.py              # BIRDeepLoader (WAV), BIRDeepImageLoader (PNG + bbox)
    ├── image_folder_loader.py         # ImageFolderLoader — class-per-subfolder
    ├── text_loader.py                 # TextFolderLoader, TextJSONLoader, TextCSVLoader
    ├── tabular_loader.py              # TabularLoader — CSV/TSV/JSON/Parquet/Excel/HDF5/SQLite
    └── video_folder_loader.py         # VideoFolderLoader — class-per-subfolder .mp4/.avi/…

config/
└── feature_extraction.yaml            # Example multi-run config (edit & run with --config)
```

---

## Conventions

### Adding a new extractor

1. Create `src/preprocessing/feature_extraction/<modality>/<file>.py`.
2. Define class with **class-level** `name`, `feature_type`, `modality`.
3. Decorate with `@register` (import from `..registry`).
4. Implement `extract(sample_path, **kwargs) → np.ndarray`.
   - Override `extract_dataset()` only if corpus-level fitting is needed.
5. Add to `feature_extraction/__init__.py` (import + `__all__`).

### Adding a new loader

1. Create `src/preprocessing/dataset_loaders/<name>_loader.py`.
2. Subclass `BaseDatasetLoader`; implement `__iter__` and `__len__`.
3. `__iter__` yields `(Optional[Path], Optional[str], dict)`.
4. Export from `dataset_loaders/__init__.py`.
5. Add a branch in `pipeline.py → _build_loader()`.

### Persistence layout (`data/processed/<run_name>/`)

```
features.npy          float32  (N, *feature_dims)
labels.npy            int32    (N,)           — omitted if unsupervised
label_names.json      list[str]
cluster_assignments.npy int32  (N,)           — omitted if not clustered
metadata.json         list[dict]
info.json             {feature_type, modality, n_samples, feature_shape, …}
```

---

## Running the CLI

```bash
# Single run — flags
python -m src.preprocessing.pipeline \
    --loader birdeep --dataset data/raw/BIRDeep_AudioAnnotations \
    --split train --extractor audio_classical \
    --output data/processed/birdeep_classical

# Generic audio dataset (class-per-subfolder layout)
python -m src.preprocessing.pipeline \
    --loader audio_folder --audio-folder data/raw/my_audio_dataset \
    --split train --extractor audio_classical \
    --output data/processed/audio_classical

# Video (class-per-subfolder layout)
python -m src.preprocessing.pipeline \
    --loader video_folder --video-folder data/raw/my_videos \
    --split train --extractor video_classical \
    --output data/processed/video_classical

# Multi-run batch — YAML config
python -m src.preprocessing.pipeline --config config/feature_extraction.yaml
```

---

## Pending Work

### Stage 3 - Model training

Needs refactoring to implement multiple ML models: from classical (SVM, LDA, PCA, Decision Tree, Random Forest, K-NN, K-Means) through deep learning (RNN, CNN, Transformers).

### Stage 4 - Model evaluation

Currently lives inside training.py, effectively compacting Stages 3 & 4 together. This is fine. Investigate whether MLFlow is still the best option or if some form of local storage is better than a dockerized container.

### Stage 5 - Model selection

Should read the necessary metrics from MLFlow to select the best model according to multiple parameters, respecting the end goal of optimizing and compiling to an embedded device with limited resources.

### Stage 10 — Model updating

Not yet designed. Will need a feedback loop from the monitoring dashboard
back to the training stage.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| librosa | Audio loading, MFCCs, spectrograms, CQT |
| scikit-image | HOG, LBP, GLCM (image + video classical) |
| scikit-learn | ColumnTransformer, scalers, PolynomialFeatures |
| opencv-python | Video I/O and optical flow (VideoCapture, Farneback) |
| tensorflow | MobileNetV2 (image/video deep), model training |
| tensorflow-metal | Apple Silicon GPU backend |
| sentence-transformers | all-MiniLM-L6-v2 sentence embeddings |
| transformers | BERT tokenizer for TextBERTTokens |
| pandas + pyarrow | Tabular loading (Parquet, Feather, JSONL) |
| openpyxl | Excel support for TabularLoader |
| h5py | HDF5 support for TabularLoader |
| PyYAML | YAML config parsing (`config.py`) |
| mlflow | Experiment tracking (Stage 4) |

---

## Environment Notes

- Activate the venv before running anything.
  If Pylance shows "cannot be resolved", run **Python: Select Interpreter**.
- Apple Silicon: `tensorflow-metal` is in `requirements.txt`.
- Do **not** commit `.npy` / `.json` processed data — large and reproducible.
- `pip install opencv-python` if video extractors raise `ModuleNotFoundError`.