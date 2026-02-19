"""
Tabular dataset loader — multi-format.

Reads an entire structured dataset from a single file and yields one row
per sample as ``(None, label, row_dict)``.  The *None* path signals to
tabular feature extractors that data lives in the metadata dict, not on disk.

Supported formats
-----------------
Format      Extension(s)             Backend
---------   -----------------------  --------------------------------
CSV / TSV   .csv, .tsv               pandas
JSON        .json                    pandas (records / split / table)
JSONL       .jsonl, .ndjson          pandas (lines=True)
Parquet     .parquet, .pq            pandas + pyarrow  (in deps)
Arrow/Feather .arrow, .feather       pandas + pyarrow  (in deps)
Excel       .xlsx, .xls, .xlsm       pandas + openpyxl (added to deps)
HDF5        .h5, .hdf5, .hdf         pandas + h5py     (in deps)
SQLite      .db, .sqlite, .sqlite3   pandas + stdlib sqlite3

Usage
-----
::

    from src.preprocessing.dataset_loaders.tabular_loader import TabularLoader
    from src.preprocessing.feature_extraction import get

    loader    = TabularLoader("data/titanic.csv", label_col="Survived")
    extractor = get("tabular_classical")()
    fs = extractor.extract_dataset(loader)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Map file suffixes → loader strategy
_FORMAT_MAP: dict[str, str] = {
    ".csv":     "csv",
    ".tsv":     "csv",
    ".txt":     "csv",
    ".json":    "json",
    ".jsonl":   "jsonl",
    ".ndjson":  "jsonl",
    ".parquet": "parquet",
    ".pq":      "parquet",
    ".arrow":   "feather",
    ".feather": "feather",
    ".xlsx":    "excel",
    ".xls":     "excel",
    ".xlsm":    "excel",
    ".xlsb":    "excel",
    ".h5":      "hdf",
    ".hdf5":    "hdf",
    ".hdf":     "hdf",
    ".db":      "sqlite",
    ".sqlite":  "sqlite",
    ".sqlite3": "sqlite",
}


class TabularLoader(BaseDatasetLoader):
    """Load a tabular dataset from a file and yield rows as metadata dicts.

    Each call to ``__iter__`` yields
    ``(None, label_value_or_None, {col: val, …})``.

    The *None* path indicates that data is in-memory; tabular feature
    extractors read from ``**kwargs`` rather than from a file.

    Parameters
    ----------
    path:
        Path to the data file.  Format is auto-detected from the suffix;
        override with *format*.
    label_col:
        Name (or 0-based index) of the column to use as the class label.
        *None* → all labels will be *None* (unsupervised).
    format:
        Explicit format string (``"csv"``, ``"json"``, ``"jsonl"``,
        ``"parquet"``, ``"feather"``, ``"excel"``, ``"hdf"``,
        ``"sqlite"``).  *None* → auto-detect from suffix.
    sheet_name:
        For Excel files: sheet name or 0-based index (default ``0``).
    hdf_key:
        For HDF5 files: the store key (default ``"data"``).
    sqlite_table:
        For SQLite files: table or view name.  *None* → first table found.
    sql_query:
        Raw SQL query string (overrides *sqlite_table*).
    read_kwargs:
        Extra keyword arguments forwarded to the underlying ``pandas``
        read function (e.g. ``sep="|"`` for CSV, ``engine="pyarrow"`` for
        Parquet).
    drop_cols:
        Column names to discard before yielding rows.
    max_rows:
        Maximum number of rows to load (useful for quick tests).
    """

    def __init__(
        self,
        path:          Path | str,
        label_col:     Optional[str | int] = None,
        format:        Optional[str]       = None,  # noqa: A002
        sheet_name:    str | int           = 0,
        hdf_key:       str                 = "data",
        sqlite_table:  Optional[str]       = None,
        sql_query:     Optional[str]       = None,
        read_kwargs:   Optional[dict]      = None,
        drop_cols:     Optional[list[str]] = None,
        max_rows:      Optional[int]       = None,
    ) -> None:
        self._path         = Path(path)
        self._label_col    = label_col
        self._sheet_name   = sheet_name
        self._hdf_key      = hdf_key
        self._sqlite_table = sqlite_table
        self._sql_query    = sql_query
        self._read_kwargs  = read_kwargs or {}
        self._drop_cols    = drop_cols or []
        self._fmt          = format or self._detect_format()

        self._df = self._load(max_rows)
        logger.info(
            "TabularLoader [%s]: %d rows × %d columns from %s.",
            self._fmt, len(self._df), self._df.shape[1], self._path.name,
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[tuple[None, Optional[str], dict]]:
        label_col = self._resolve_label_col()
        for _, row in self._df.iterrows():
            label = str(row[label_col]) if label_col is not None else None
            meta  = {
                k: (None if pd.isna(v) else v)
                for k, v in row.items()
                if k != label_col
            }
            yield None, label, meta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_format(self) -> str:
        suffix = self._path.suffix.lower()
        fmt = _FORMAT_MAP.get(suffix)
        if fmt is None:
            raise ValueError(
                f"Cannot auto-detect format for suffix {suffix!r}. "
                f"Supported: {sorted(set(_FORMAT_MAP.values()))}. "
                "Pass format= explicitly."
            )
        return fmt

    def _resolve_label_col(self) -> Optional[str]:
        if self._label_col is None:
            return None
        if isinstance(self._label_col, int):
            return self._df.columns[self._label_col]
        if self._label_col in self._df.columns:
            return self._label_col
        logger.warning(
            "label_col %r not found in columns %s; ignoring.",
            self._label_col, list(self._df.columns),
        )
        return None

    def _load(self, max_rows: Optional[int]) -> pd.DataFrame:
        kw = self._read_kwargs.copy()
        fmt = self._fmt

        if fmt == "csv":
            df = pd.read_csv(
                self._path,
                nrows=max_rows,
                on_bad_lines="warn",
                **kw,
            )
        elif fmt == "json":
            df = pd.read_json(self._path, **kw)
            if max_rows:
                df = df.head(max_rows)
        elif fmt == "jsonl":
            df = pd.read_json(self._path, lines=True, nrows=max_rows, **kw)
        elif fmt == "parquet":
            df = pd.read_parquet(self._path, **kw)
            if max_rows:
                df = df.head(max_rows)
        elif fmt == "feather":
            df = pd.read_feather(self._path, **kw)
            if max_rows:
                df = df.head(max_rows)
        elif fmt == "excel":
            df = pd.read_excel(
                self._path,
                sheet_name=self._sheet_name,
                nrows=max_rows,
                **kw,
            )
        elif fmt == "hdf":
            df = pd.read_hdf(self._path, key=self._hdf_key, **kw)
            if max_rows:
                df = df.head(max_rows)
        elif fmt == "sqlite":
            df = self._load_sqlite(max_rows, kw)
        else:
            raise ValueError(f"Unsupported format: {fmt!r}")

        # Drop unwanted columns
        cols_to_drop = [c for c in self._drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df.reset_index(drop=True)

    def _load_sqlite(self, max_rows: Optional[int], kw: dict) -> pd.DataFrame:
        con = sqlite3.connect(self._path)
        try:
            if self._sql_query:
                query = self._sql_query
            else:
                table = self._sqlite_table or self._first_table(con)
                limit = f" LIMIT {max_rows}" if max_rows else ""
                query = f'SELECT * FROM "{table}"{limit}'
            df = pd.read_sql_query(query, con, **kw)
        finally:
            con.close()
        return df

    @staticmethod
    def _first_table(con: sqlite3.Connection) -> str:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cur.fetchall()]
        if not tables:
            raise RuntimeError("No tables found in the SQLite database.")
        return tables[0]