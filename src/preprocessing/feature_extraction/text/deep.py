"""
Deep text feature extractors.

Preserves semantic or sequential structure for deep learning models.

Extractor               name                    Output shape   Suited for
----------------------  ----------------------  -------------  -----------------------
TextSentenceEmbedding   text_sentence_embed     (D,)           Semantic similarity,
                                                               SVM/KNN on embeddings,
                                                               fine-tuned Transformer
TextBERTTokens          text_bert_tokens        (max_length,)  RNN, LSTM,
                                                               Transformer from scratch

Both extractors are stateless (pre-trained vocabularies / models) and work
sample-by-sample via the standard ``extract()`` interface — no corpus-level
fitting required.

Dependencies
------------
``sentence-transformers`` — required for ``TextSentenceEmbedding``.
``transformers``          — required for ``TextBERTTokens`` (installed as a
                            transitive dependency of sentence-transformers).

Both are imported lazily on first use to avoid slow import times when only
the classical extractors are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseFeatureExtractor
from ..registry import register
from .classical import _read_text  # shared helper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence embeddings (sentence-transformers)
# ---------------------------------------------------------------------------

@register
class TextSentenceEmbedding(BaseFeatureExtractor):
    """Dense sentence-level embedding via a pretrained Transformer encoder.

    Uses ``sentence-transformers`` to encode each document into a
    fixed-size dense vector.  The default model ``all-MiniLM-L6-v2``
    (384 dimensions, ~22 MB) strikes a good balance between quality and
    inference speed.

    Shape: ``(embedding_dim,)`` — 384 for the default model.

    This representation is suitable for:
    - Semantic similarity and nearest-neighbour search (K-NN)
    - SVM / logistic regression on top of frozen embeddings
    - Fine-tuning a Transformer classifier head

    Parameters
    ----------
    model_name:
        Any model ID from the sentence-transformers model hub, e.g.:
        ``"all-MiniLM-L6-v2"``      (384-dim, fast, general purpose)
        ``"all-mpnet-base-v2"``      (768-dim, higher quality)
        ``"paraphrase-multilingual-MiniLM-L12-v2"``  (multilingual)
    device:
        ``"cpu"`` or ``"cuda"`` (or ``"mps"`` on Apple Silicon).
        *None* → auto-detected.
    batch_size:
        Number of documents encoded per forward pass in ``extract_dataset()``.
        Has no effect on single-document ``extract()`` calls.
    normalize_embeddings:
        If *True*, L2-normalise the output vectors.  Recommended for cosine-
        similarity–based downstream tasks.
    """

    name         = "text_sentence_embed"
    feature_type = "deep"
    modality     = "text"

    def __init__(
        self,
        model_name:           str            = "all-MiniLM-L6-v2",
        device:               Optional[str]  = None,
        batch_size:           int            = 64,
        normalize_embeddings: bool           = True,
    ) -> None:
        self.model_name           = model_name
        self.device               = device
        self.batch_size           = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model               = None  # lazy init

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        """Encode a single document and return its embedding vector."""
        text = _read_text(sample_path, kwargs)
        model = self._get_model()
        vec = model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for TextSentenceEmbedding. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading sentence-transformer model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


# ---------------------------------------------------------------------------
# BERT / HuggingFace tokeniser → token ID sequences
# ---------------------------------------------------------------------------

@register
class TextBERTTokens(BaseFeatureExtractor):
    """BERT-style token ID sequence for RNN / LSTM / Transformer models.

    Uses a HuggingFace ``AutoTokenizer`` to convert raw text into a
    fixed-length integer sequence of sub-word token IDs.  The output is
    directly usable as input to any HuggingFace model, a custom LSTM, or
    a Transformer trained from scratch.

    Shape: ``(max_length,)`` — padded / truncated to ``max_length`` tokens.

    The vocabulary is pre-built (it comes with the pretrained tokeniser);
    no corpus-level fitting is required.

    Parameters
    ----------
    model_name:
        Any HuggingFace tokeniser name, e.g.:
        ``"bert-base-uncased"``
        ``"distilbert-base-uncased"``
        ``"roberta-base"``
        ``"xlm-roberta-base"``    (multilingual)
    max_length:
        Sequence length after padding / truncation.
    return_attention_mask:
        If *True*, the extractor returns a ``(2, max_length)`` array where
        row 0 is token IDs and row 1 is the attention mask.  If *False*
        (default), returns only the ``(max_length,)`` token ID vector.
    """

    name         = "text_bert_tokens"
    feature_type = "deep"
    modality     = "text"

    def __init__(
        self,
        model_name:            str  = "distilbert-base-uncased",
        max_length:            int  = 512,
        return_attention_mask: bool = False,
    ) -> None:
        self.model_name            = model_name
        self.max_length            = max_length
        self.return_attention_mask = return_attention_mask
        self._tokenizer            = None  # lazy init

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        """Tokenise a single document and return its token ID sequence."""
        text = _read_text(sample_path, kwargs)
        tok = self._get_tokenizer()
        enc = tok(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ids = enc["input_ids"][0].astype(np.int32)  # (max_length,)
        if self.return_attention_mask:
            mask = enc["attention_mask"][0].astype(np.int32)
            return np.stack([ids, mask])  # (2, max_length)
        return ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TextBERTTokens. "
                "It is installed automatically with sentence-transformers."
            ) from exc
        logger.info("Loading HuggingFace tokenizer: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer