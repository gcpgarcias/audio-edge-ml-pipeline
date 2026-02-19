# CLAUDE.md — Audio Edge ML Pipeline

This file is read automatically by Claude Code at the start of every session.
Keep it concise (< 200 lines); link to topic files in `.claude/` for details.

---

## Project Overview

A **10-stage ML pipeline** for deploying species-classification models on embedded
devices (Arduino Nicla Vision). Primary dataset: **BIRDeep_AudioAnnotations**
(36 bird species, WAV + PNG pairs, YOLOv8 bbox annotations, pre-split CSVs).

### Pipeline Stages

| # | Stage | Language / Tool | Status |
|---|-------|----------------|--------|
| 1 | Data ingestion | Python / FastAPI (`src/ingestion/api.py`) | Done |
| 2 | Data transformation | Python | In progress |
| 3 | Model training | TensorFlow (`src/training/`) | Needs expansion |
| 4 | Model evaluation | MLflow (`docker/`) | Needs expansion |
| 5 | Model selection | scikit-learn | TODO |
| 6 | Model optimization | LiteRT (`src/optimization/quantize.py`) | Needs expansion |
| 7 | Model compilation | Apache TVM | TODO |
| 8 | Model deployment | PlatformIO (`src/deployment/edge_simulator.py`) | TODO |
| 9 | Model monitoring | Streamlit (`src/monitoring/dashboard.py`) | Needs refactoring |
| 10 | Model updating | — | TODO |

---

## Stage 2 Architecture — Feature Extraction

All feature extraction lives in `src/preprocessing/`. The design follows a
**modality × feature-type** grid with a registry/factory pattern.

### Key Abstractions (`src/preprocessing/feature_extraction/base.py`)

- **`FeatureSet`** — uniform output container; `labels` and `cluster_assignments`
  are both `Optional[np.ndarray]` to support supervised, unsupervised, and
  semi-supervised workflows (use `-1` in labels for unlabelled samples).
- **`BaseFeatureExtractor`** — ABC with `extract(Optional[Path], **kwargs)`
  and `extract_dataset(loader, max_samples) → FeatureSet`. `sample_path` is
  `None` for in-memory data (tabular rows, JSON text).
- **`BaseDatasetLoader`** — ABC yielding `(Optional[Path], Optional[str], dict)`
  triples; `None` path signals in-memory data.

### Registry (`src/preprocessing/feature_extraction/registry.py`)

```python
# Register a new extractor with a single decorator:
@register
class MyExtractor(BaseFeatureExtractor):
    name         = "my_extractor"   # must be globally unique
    feature_type = "classical"      # "classical" | "deep"
    modality     = "audio"          # "audio" | "image" | "text" | "tabular" | "video"
```

The registry is populated automatically when any class inside
`feature_extraction/` is imported — the `__init__.py` files do this.
`get("my_extractor")` returns the class; `list_extractors()` lists all names.

### Registered Extractors (15 total)

```
audio_classical   audio_mel_spec   audio_waveform   audio_cqt   audio_mfcc_seq
image_classical   image_pixels     image_mobilenet_v2
text_tfidf        text_bow         text_char_ngram  text_sentence_embed  text_bert_tokens
tabular_classical tabular_polynomial
```

### Stateful Extractors (fit before transform)

Text classical (TF-IDF, BoW, char n-gram) and tabular extractors need a
corpus-level fit. They **override `extract_dataset()`** to do fit+transform in
one pass. Calling `extract()` before `extract_dataset()` raises `RuntimeError`.

---

## File Map

```
src/preprocessing/
├── pipeline.py                        # FeaturePipeline orchestrator + CLI
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
│   └── tabular/
│       ├── classical.py               # TabularClassicalExtractor — ColumnTransformer pipeline
│       └── deep.py                    # TabularPolynomialExtractor — degree-2 poly expansion
└── dataset_loaders/
    ├── birdeep_loader.py              # BIRDeepLoader (WAV), BIRDeepImageLoader (PNG + bbox)
    ├── image_folder_loader.py         # ImageFolderLoader — class-per-subfolder
    ├── text_loader.py                 # TextFolderLoader, TextJSONLoader, TextCSVLoader
    └── tabular_loader.py              # TabularLoader — CSV/TSV/JSON/Parquet/Excel/HDF5/SQLite
```

---

## Conventions

### Adding a new extractor

1. Create `src/preprocessing/feature_extraction/<modality>/<file>.py`.
2. Define the class with **class-level** `name`, `feature_type`, `modality`.
3. Decorate with `@register` (import from `..registry`).
4. Implement `extract(sample_path, **kwargs) → np.ndarray`.
   - Override `extract_dataset()` only if fitting requires the full corpus.
5. Add the class to `feature_extraction/__init__.py` (both import and `__all__`).

### Adding a new loader

1. Create `src/preprocessing/dataset_loaders/<name>_loader.py`.
2. Subclass `BaseDatasetLoader`; implement `__iter__` and `__len__`.
3. `__iter__` must yield `(Optional[Path], Optional[str], dict)`.
   - Use `None` for path when data is in-memory.
4. Export from `dataset_loaders/__init__.py`.
5. Wire into the CLI in `pipeline.py → main()`.

### Persistence (FeaturePipeline.save / .load)

Saved artefacts land in `data/processed/<run_name>/`:
```
features.npy         float32  (N, *feature_dims)
labels.npy           int32    (N,)              — omitted when None
label_names.json     list[str]
cluster_assignments.npy int32 (N,)              — omitted when None
metadata.json        list[dict]
info.json            {feature_type, modality, n_samples, feature_shape, …}
```

---

## Running the CLI

```bash
# Classical audio features from BIRDeep (training split)
python -m src.preprocessing.pipeline \
    --loader birdeep --dataset data/raw/BIRDeep_AudioAnnotations \
    --split train --extractor audio_classical \
    --output data/processed/birdeep_classical

# MobileNetV2 image embeddings (validation split)
python -m src.preprocessing.pipeline \
    --loader birdeep_image --dataset data/raw/BIRDeep_AudioAnnotations \
    --split validation --extractor image_mobilenet_v2 \
    --output data/processed/birdeep_mobilenet

# Tabular CSV
python -m src.preprocessing.pipeline \
    --loader tabular --dataset data/raw/my_data.csv \
    --label-col target_column \
    --extractor tabular_classical \
    --output data/processed/tabular_run
```

---

## Pending Work

### Phase 8 — Video features (not started)

- `src/preprocessing/feature_extraction/video/classical.py`
  — per-frame HOG/histogram aggregation, optical flow statistics
- `src/preprocessing/feature_extraction/video/deep.py`
  — frame sequence tensors `(T, H, W, C)`, MobileNetV2-per-frame embeddings
- `src/preprocessing/dataset_loaders/video_folder_loader.py`
  — class-per-subfolder .mp4/.avi loader using OpenCV or decord

### Phase 9 — Pipeline orchestrator enhancements

- `pipeline.py` CLI currently has no `--config` flag for YAML-driven runs.
- Consider adding a config schema for reproducible, multi-extractor experiments.

### Phase 10 — Model updating

- Not yet designed. Will need a feedback loop from the monitoring dashboard
  back to the training stage.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| librosa | Audio loading, MFCCs, spectrograms, CQT |
| scikit-image | HOG, LBP, GLCM (image classical) |
| scikit-learn | ColumnTransformer, scalers, PolynomialFeatures |
| tensorflow | MobileNetV2 (image deep), model training |
| tensorflow-metal | Apple Silicon GPU backend |
| sentence-transformers | all-MiniLM-L6-v2 sentence embeddings |
| transformers | BERT tokenizer for TextBERTTokens |
| pandas + pyarrow | Tabular loading (Parquet, Feather, JSONL) |
| openpyxl | Excel support for TabularLoader |
| h5py | HDF5 support for TabularLoader |
| mlflow | Experiment tracking (Stage 4) |

---

## Environment Notes

- Python environment: activate the venv before running anything.
  The `.vscode/settings.json` should point to the correct interpreter.
  If Pylance shows "cannot be resolved" for installed packages, run
  **Python: Select Interpreter** in VSCode and pick the venv.
- Apple Silicon: `tensorflow-metal` is already in `requirements.txt`.
- Do **not** commit `.npy` / `.json` processed data files — they are large
  and reproducible from the raw dataset.