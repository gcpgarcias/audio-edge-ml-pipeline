"""
Text dataset loaders.

Three loaders cover the most common corpus layouts:

``TextFolderLoader``
    Class-per-subfolder tree where each ``.txt`` file is one document:
    ``<root>/<class_name>/<doc_001>.txt``

``TextJSONLoader``
    JSON file (or newline-delimited JSONL) where each record has a text
    field and an optional label field.  Documents are held in memory; the
    loader yields ``(None, label, {"text": "...", ...})``.

``TextCSVLoader``
    CSV / TSV file with a configurable text column and an optional label
    column.  Also yields in-memory ``(None, label, {"text": "...", ...})``.

All loaders are compatible with both file-path–based extractors (which read
``sample_path``) and in-memory extractors (which read ``kwargs["text"]``).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".text", ".md"})


# ---------------------------------------------------------------------------
# TextFolderLoader — class-per-folder, one .txt per sample
# ---------------------------------------------------------------------------

class TextFolderLoader(BaseDatasetLoader):
    """Load text documents from a class-per-subfolder directory tree.

    Parameters
    ----------
    root:
        Dataset root (or ``<root>/<split>/`` when *split* is given).
    split:
        Optional split subdirectory name.
    extensions:
        File extensions to include (case-insensitive).  Defaults to
        ``{".txt", ".text", ".md"}``.
    class_names:
        If given, only documents in these class folders are loaded, in this
        order.  *None* → all subdirectories, sorted alphabetically.
    encoding:
        Text file encoding (default ``"utf-8"``).
    """

    def __init__(
        self,
        root:        Path | str,
        split:       Optional[str]       = None,
        extensions:  Optional[set[str]]  = None,
        class_names: Optional[list[str]] = None,
        encoding:    str                 = "utf-8",
    ) -> None:
        effective_root = Path(root) / split if split else Path(root)
        if not effective_root.is_dir():
            raise NotADirectoryError(f"Dataset root not found: {effective_root}")

        self._encoding   = encoding
        self._extensions = (
            frozenset(e.lower() for e in extensions)
            if extensions is not None
            else _TEXT_SUFFIXES
        )

        if class_names is not None:
            self._class_names = list(class_names)
            class_dirs = [effective_root / c for c in class_names]
        else:
            class_dirs = sorted(p for p in effective_root.iterdir() if p.is_dir())
            self._class_names = [d.name for d in class_dirs]

        self._samples: list[tuple[Path, str]] = []
        for class_dir, label in zip(class_dirs, self._class_names):
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s (skipping)", class_dir)
                continue
            docs = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._extensions
            )
            for path in docs:
                self._samples.append((path, label))

        logger.info(
            "TextFolderLoader: %d documents across %d classes.",
            len(self._samples), len(self._class_names),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[tuple[Optional[Path], Optional[str], dict]]:
        for path, label in self._samples:
            yield path, label, {"filename": path.name, "class_dir": path.parent.name}

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


# ---------------------------------------------------------------------------
# TextJSONLoader — JSON / JSONL in-memory corpus
# ---------------------------------------------------------------------------

class TextJSONLoader(BaseDatasetLoader):
    """Load a text corpus from a JSON or JSONL file.

    Supported layouts:

    - **JSON array**: ``[{"text": "...", "label": "cat"}, ...]``
    - **JSONL** (newline-delimited): one JSON object per line.
    - **Single object with a records key**:
      ``{"data": [{"text": "...", "label": "..."}]}``

    Documents are held in memory; the loader yields
    ``(None, label, {"text": "...", <extra fields>})``.

    Parameters
    ----------
    path:
        Path to the JSON / JSONL file.
    text_key:
        Record field containing the document text.
    label_key:
        Record field containing the class label.  *None* → unsupervised.
    records_key:
        If the JSON root is a dict, the key under which the list of records
        lives.  *None* → the root itself must be a list (or JSONL).
    """

    def __init__(
        self,
        path:        Path | str,
        text_key:    str            = "text",
        label_key:   Optional[str]  = "label",
        records_key: Optional[str]  = None,
    ) -> None:
        self._path        = Path(path)
        self._text_key    = text_key
        self._label_key   = label_key
        self._records_key = records_key
        self._records     = self._load()
        logger.info("TextJSONLoader: %d records from %s.", len(self._records), path)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[tuple[Optional[Path], Optional[str], dict]]:
        for rec in self._records:
            text  = str(rec.get(self._text_key, ""))
            label = str(rec[self._label_key]) if self._label_key and self._label_key in rec else None
            meta  = {k: v for k, v in rec.items() if k != self._label_key}
            meta["text"] = text
            yield None, label, meta

    # ------------------------------------------------------------------
    def _load(self) -> list[dict]:
        text = self._path.read_text(encoding="utf-8", errors="replace")
        # Try JSONL first
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            try:
                records = [json.loads(l) for l in lines]
                if all(isinstance(r, dict) for r in records):
                    return records
            except json.JSONDecodeError:
                pass
        # Fall back to standard JSON
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            key = self._records_key or next(
                (k for k, v in obj.items() if isinstance(v, list)), None
            )
            if key and isinstance(obj.get(key), list):
                return obj[key]
        raise ValueError(
            f"Cannot parse {self._path} as a JSON array or JSONL corpus. "
            "Expected a list of records or an object with a list field."
        )


# ---------------------------------------------------------------------------
# TextCSVLoader — CSV / TSV in-memory corpus
# ---------------------------------------------------------------------------

class TextCSVLoader(BaseDatasetLoader):
    """Load a text corpus from a CSV or TSV file.

    Documents are held in memory; the loader yields
    ``(None, label, {"text": "...", <extra fields>})``.

    Parameters
    ----------
    path:
        Path to the CSV / TSV file.
    text_col:
        Column name (or 0-based index) containing the document text.
    label_col:
        Column name (or 0-based index) containing the class label.
        *None* → unsupervised (all labels will be *None*).
    delimiter:
        Field delimiter.  *None* → sniffed from the first 8 KB.
    encoding:
        File encoding (default ``"utf-8"``).
    skip_header:
        If *True* (default), treat the first row as a header.
    """

    def __init__(
        self,
        path:        Path | str,
        text_col:    str | int      = "text",
        label_col:   str | int | None = "label",
        delimiter:   Optional[str]  = None,
        encoding:    str            = "utf-8",
        skip_header: bool           = True,
    ) -> None:
        self._path      = Path(path)
        self._text_col  = text_col
        self._label_col = label_col
        self._encoding  = encoding
        self._rows      = self._load(delimiter, skip_header)
        logger.info("TextCSVLoader: %d rows from %s.", len(self._rows), path)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[tuple[Optional[Path], Optional[str], dict]]:
        for row in self._rows:
            yield None, row["_label"], {k: v for k, v in row.items() if k != "_label"}

    # ------------------------------------------------------------------
    def _load(self, delimiter: Optional[str], skip_header: bool) -> list[dict]:
        raw_text = self._path.read_text(encoding=self._encoding, errors="replace")

        if delimiter is None:
            sample = raw_text[:8192]
            try:
                dialect  = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        reader = csv.reader(raw_text.splitlines(), delimiter=delimiter)
        rows_raw = list(reader)
        if not rows_raw:
            return []

        if skip_header:
            header    = rows_raw[0]
            data_rows = rows_raw[1:]
        else:
            header    = [str(i) for i in range(len(rows_raw[0]))]
            data_rows = rows_raw

        # Resolve column identifiers
        def _resolve(col: str | int) -> int:
            if isinstance(col, int):
                return col
            if col in header:
                return header.index(col)
            raise KeyError(
                f"Column {col!r} not found in CSV header: {header}. "
                "Set text_col / label_col to the 0-based column index."
            )

        text_idx  = _resolve(self._text_col)
        label_idx = _resolve(self._label_col) if self._label_col is not None else None

        records = []
        for row in data_rows:
            if not row:
                continue
            meta = {header[i]: row[i] for i in range(min(len(header), len(row)))}
            meta["text"] = row[text_idx] if text_idx < len(row) else ""
            meta["_label"] = (
                row[label_idx] if label_idx is not None and label_idx < len(row)
                else None
            )
            records.append(meta)
        return records