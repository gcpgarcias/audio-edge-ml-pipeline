"""
Classical text feature extractors.

Produces fixed-length flat feature vectors from raw text, suitable for:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Extractor              name            Vectorizer           Notes
---------------------  --------------  -------------------  --------------------
TextTFIDFExtractor     text_tfidf      TfidfVectorizer      word n-grams
TextBoWExtractor       text_bow        CountVectorizer      word n-grams
TextCharNgramExtractor text_char_ngram TfidfVectorizer      char n-grams (wb)

Design note — fit-before-transform
-----------------------------------
TF-IDF and BoW require a vocabulary fitted on the full corpus before any
single document can be transformed.  These extractors therefore override
``extract_dataset()`` to perform a single-pass fit+transform over all
documents yielded by the loader.

Calling ``extract()`` directly is supported *after* ``extract_dataset()``
has been called at least once (the fitted vectorizer is cached).  This is
useful for transforming held-out documents with the training vocabulary.

Text sources
------------
Extractors read text from either:
- a ``Path`` pointing to a UTF-8 encoded plain-text file, or
- a ``text`` key in the metadata dict (set by JSON/CSV loaders that hold
  documents in memory rather than on disk).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..base import BaseDatasetLoader, BaseFeatureExtractor, FeatureSet
from ..registry import register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_text(sample_path: Optional[Path], kwargs: dict) -> str:
    """Return the text string for a sample.

    Priority:
    1. ``kwargs["text"]``  – in-memory text from JSON/CSV loaders.
    2. ``sample_path``     – read UTF-8 from file.
    """
    if "text" in kwargs:
        return str(kwargs["text"])
    if sample_path is not None:
        return Path(sample_path).read_text(encoding="utf-8", errors="replace")
    raise ValueError("Neither 'text' in metadata nor a valid sample_path was provided.")


def _build_feature_set(
    extractor: BaseFeatureExtractor,
    X: np.ndarray,
    raw_labels: list[Optional[str]],
    all_meta: list[dict],
) -> FeatureSet:
    """Assemble a FeatureSet from an already-vectorised matrix."""
    label_to_idx: dict[str, int] = {}
    all_labels: list[int] = []
    for lbl in raw_labels:
        if lbl is not None:
            if lbl not in label_to_idx:
                label_to_idx[lbl] = len(label_to_idx)
            all_labels.append(label_to_idx[lbl])

    labels = np.array(all_labels, dtype=np.int32) if all_labels else None
    label_names = (
        [k for k, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
        if label_to_idx else None
    )
    return FeatureSet(
        features=X,
        feature_type=extractor.feature_type,
        modality=extractor.modality,
        metadata=all_meta,
        labels=labels,
        label_names=label_names,
    )


def _collect_corpus(
    loader: BaseDatasetLoader,
    max_samples: Optional[int],
) -> tuple[list[str], list[Optional[str]], list[dict]]:
    """One-pass corpus collection from a loader."""
    texts, labels, metas = [], [], []
    for i, (path, label, meta) in enumerate(loader):
        if max_samples is not None and i >= max_samples:
            break
        try:
            texts.append(_read_text(path, meta))
            labels.append(label)
            metas.append(meta)
        except Exception as exc:
            logger.warning("Skipping sample %d: %s", i, exc)
    return texts, labels, metas


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

@register
class TextTFIDFExtractor(BaseFeatureExtractor):
    """TF-IDF weighted bag-of-words (word n-grams).

    Parameters
    ----------
    max_features:
        Vocabulary size cap (most frequent terms).
    ngram_range:
        ``(min_n, max_n)`` word n-gram range.
    sublinear_tf:
        Apply ``1 + log(tf)`` term-frequency scaling.
    min_df, max_df:
        Minimum / maximum document frequency thresholds.
    """

    name         = "text_tfidf"
    feature_type = "classical"
    modality     = "text"

    def __init__(
        self,
        max_features: int         = 10_000,
        ngram_range:  tuple       = (1, 2),
        sublinear_tf: bool        = True,
        min_df:       int | float = 2,
        max_df:       float       = 0.95,
    ) -> None:
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            strip_accents="unicode",
            decode_error="replace",
        )
        self._fitted = False

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "Vocabulary not fitted. Call extract_dataset() on a training "
                "corpus first."
            )
        text = _read_text(sample_path, kwargs)
        return self._vec.transform([text]).toarray()[0].astype(np.float32)

    def extract_dataset(
        self,
        loader: BaseDatasetLoader,
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        texts, raw_labels, metas = _collect_corpus(loader, max_samples)
        if not texts:
            raise RuntimeError("No texts were successfully loaded.")
        X = self._vec.fit_transform(texts).toarray().astype(np.float32)
        self._fitted = True
        logger.info(
            "TextTFIDFExtractor: fitted on %d docs, vocabulary size %d.",
            len(texts), len(self._vec.vocabulary_),
        )
        return _build_feature_set(self, X, raw_labels, metas)

    @property
    def vocabulary_size(self) -> Optional[int]:
        return len(self._vec.vocabulary_) if self._fitted else None


# ---------------------------------------------------------------------------
# Bag of Words
# ---------------------------------------------------------------------------

@register
class TextBoWExtractor(BaseFeatureExtractor):
    """Raw term-count bag-of-words (word n-grams).

    Produces integer count vectors (cast to float32) without TF-IDF
    weighting.  Useful when downstream models benefit from raw frequencies
    (e.g. Naïve Bayes, or when you want to apply TF-IDF normalisation
    manually later).

    Parameters
    ----------
    max_features:  Vocabulary size cap.
    ngram_range:   ``(min_n, max_n)`` word n-gram range.
    binary:        If *True*, mark presence/absence rather than counts.
    min_df, max_df: Document frequency thresholds.
    """

    name         = "text_bow"
    feature_type = "classical"
    modality     = "text"

    def __init__(
        self,
        max_features: int         = 10_000,
        ngram_range:  tuple       = (1, 1),
        binary:       bool        = False,
        min_df:       int | float = 2,
        max_df:       float       = 0.95,
    ) -> None:
        self._vec = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            binary=binary,
            min_df=min_df,
            max_df=max_df,
            strip_accents="unicode",
            decode_error="replace",
        )
        self._fitted = False

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "Vocabulary not fitted. Call extract_dataset() first."
            )
        text = _read_text(sample_path, kwargs)
        return self._vec.transform([text]).toarray()[0].astype(np.float32)

    def extract_dataset(
        self,
        loader: BaseDatasetLoader,
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        texts, raw_labels, metas = _collect_corpus(loader, max_samples)
        if not texts:
            raise RuntimeError("No texts were successfully loaded.")
        X = self._vec.fit_transform(texts).toarray().astype(np.float32)
        self._fitted = True
        logger.info(
            "TextBoWExtractor: fitted on %d docs, vocabulary size %d.",
            len(texts), len(self._vec.vocabulary_),
        )
        return _build_feature_set(self, X, raw_labels, metas)


# ---------------------------------------------------------------------------
# Character n-gram TF-IDF
# ---------------------------------------------------------------------------

@register
class TextCharNgramExtractor(BaseFeatureExtractor):
    """Character-level TF-IDF n-gram features.

    Operates on character n-grams within word boundaries (``analyzer=
    'char_wb'``).  More robust than word-level features for noisy text,
    morphologically rich languages, and out-of-vocabulary terms.

    Parameters
    ----------
    max_features:  Vocabulary size cap.
    ngram_range:   ``(min_n, max_n)`` character n-gram range.
    min_df:        Minimum document frequency.
    """

    name         = "text_char_ngram"
    feature_type = "classical"
    modality     = "text"

    def __init__(
        self,
        max_features: int         = 50_000,
        ngram_range:  tuple       = (3, 5),
        min_df:       int | float = 3,
    ) -> None:
        self._vec = TfidfVectorizer(
            analyzer="char_wb",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            strip_accents="unicode",
            decode_error="replace",
        )
        self._fitted = False

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "Vocabulary not fitted. Call extract_dataset() first."
            )
        text = _read_text(sample_path, kwargs)
        return self._vec.transform([text]).toarray()[0].astype(np.float32)

    def extract_dataset(
        self,
        loader: BaseDatasetLoader,
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        texts, raw_labels, metas = _collect_corpus(loader, max_samples)
        if not texts:
            raise RuntimeError("No texts were successfully loaded.")
        X = self._vec.fit_transform(texts).toarray().astype(np.float32)
        self._fitted = True
        logger.info(
            "TextCharNgramExtractor: fitted on %d docs, vocabulary size %d.",
            len(texts), len(self._vec.vocabulary_),
        )
        return _build_feature_set(self, X, raw_labels, metas)