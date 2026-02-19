"""
Classical tabular feature extractor.

Converts a mixed-type tabular dataset (numerical + categorical columns) into
a flat, normalised feature vector using a scikit-learn ``ColumnTransformer``
pipeline.  Suitable for:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Processing pipeline per column type
-------------------------------------
Numerical columns
    1. ``SimpleImputer`` (strategy configurable, default ``"median"``)
    2. ``StandardScaler`` / ``MinMaxScaler`` / ``RobustScaler``
       (strategy configurable)

Categorical columns
    1. ``SimpleImputer`` (strategy ``"most_frequent"``)
    2. ``OneHotEncoder`` (``handle_unknown="ignore"``, ``sparse_output=False``)

Datetime columns
    Expanded into year, month, day-of-week, hour, minute components
    (all numerical) before passing through the numerical pipeline.

Column-type inference
---------------------
Column types are inferred from a pandas DataFrame built from the loader's
metadata dicts.  The inference rules are:

- ``object`` / ``category`` dtype  → categorical
- ``datetime``                      → datetime (expanded)
- ``bool``                          → categorical (binary OHE)
- anything else                     → numerical

You can override inference by passing explicit ``categorical_cols`` /
``numerical_cols`` lists to the constructor.

Design note — stateful extractor
---------------------------------
Like the text classical extractors, ``TabularClassicalExtractor`` must see
the full dataset before transforming any single row.  It therefore overrides
``extract_dataset()`` to perform a single-pass fit+transform.  After fitting,
``extract()`` accepts a row dict (passed via ``**kwargs``) and applies the
fitted transformer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from ..base import BaseDatasetLoader, BaseFeatureExtractor, FeatureSet
from ..registry import register

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


@register
class TabularClassicalExtractor(BaseFeatureExtractor):
    """Mixed-type tabular feature extractor (imputation + scaling + encoding).

    Parameters
    ----------
    numerical_cols:
        Explicit list of numerical column names.  *None* → auto-inferred.
    categorical_cols:
        Explicit list of categorical column names.  *None* → auto-inferred.
    label_col:
        Name of the target column inside the metadata dicts.  When set, this
        column is extracted as the label and removed from the feature matrix.
        Usually you rely on the loader's label rather than a column — leave
        as *None* in that case.
    scaler:
        Scaling strategy for numerical columns:
        ``"standard"`` (default), ``"minmax"``, or ``"robust"``.
    impute_numerical:
        Imputation strategy for numerical columns (``"mean"``, ``"median"``,
        ``"most_frequent"``, ``"constant"``).
    impute_categorical:
        Imputation strategy for categorical columns.
    max_ohe_categories:
        Maximum unique values a categorical column may have before it is
        dropped from OHE (prevents explosion on high-cardinality columns).
    """

    name         = "tabular_classical"
    feature_type = "classical"
    modality     = "tabular"

    def __init__(
        self,
        numerical_cols:       Optional[list[str]] = None,
        categorical_cols:     Optional[list[str]] = None,
        label_col:            Optional[str]       = None,
        scaler:               str                 = "standard",
        impute_numerical:     str                 = "median",
        impute_categorical:   str                 = "most_frequent",
        max_ohe_categories:   int                 = 50,
    ) -> None:
        if scaler not in _SCALERS:
            raise ValueError(f"scaler must be one of {list(_SCALERS)}, got {scaler!r}.")
        self._numerical_cols_arg   = numerical_cols
        self._categorical_cols_arg = categorical_cols
        self.label_col             = label_col
        self.scaler                = scaler
        self.impute_numerical      = impute_numerical
        self.impute_categorical    = impute_categorical
        self.max_ohe_categories    = max_ohe_categories
        self._transformer: Optional[ColumnTransformer] = None
        self._feature_names: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        """Transform a single row (passed via *kwargs*) using the fitted pipeline.

        Parameters
        ----------
        sample_path:
            Ignored (tabular rows have no file path).
        **kwargs:
            Column-value pairs for one row.  All columns that were present
            during fitting must be included.
        """
        if self._transformer is None:
            raise RuntimeError(
                "Transformer not fitted.  Call extract_dataset() on the "
                "training split first."
            )
        row_df = pd.DataFrame([kwargs])
        row_df = self._expand_datetimes(row_df)
        return self._transformer.transform(row_df)[0].astype(np.float32)

    def extract_dataset(
        self,
        loader: BaseDatasetLoader,
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        """Fit the ColumnTransformer on the full dataset and transform it.

        Rows are collected from the loader's metadata dicts (the ``Path``
        element is expected to be *None* for tabular loaders).
        """
        rows: list[dict]            = []
        raw_labels: list[Optional[str]] = []
        all_meta: list[dict]        = []

        for i, (_path, label, meta) in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(meta)
            raw_labels.append(label)
            all_meta.append(meta)

        if not rows:
            raise RuntimeError("No rows were yielded by the loader.")

        df = pd.DataFrame(rows)

        # Extract label from a dedicated column when requested
        if self.label_col and self.label_col in df.columns:
            col_labels = df.pop(self.label_col).astype(str).tolist()
            # Override loader labels with column values when they exist
            raw_labels = [c if c != "nan" else r for c, r in zip(col_labels, raw_labels)]

        df = self._expand_datetimes(df)

        num_cols, cat_cols = self._infer_columns(df)
        logger.info(
            "TabularClassicalExtractor: %d numerical, %d categorical columns.",
            len(num_cols), len(cat_cols),
        )

        self._transformer = self._build_transformer(num_cols, cat_cols)
        X = self._transformer.fit_transform(df).astype(np.float32)

        label_to_idx: dict[str, int] = {}
        int_labels: list[int] = []
        for lbl in raw_labels:
            if lbl is not None:
                if lbl not in label_to_idx:
                    label_to_idx[lbl] = len(label_to_idx)
                int_labels.append(label_to_idx[lbl])

        labels = np.array(int_labels, dtype=np.int32) if int_labels else None
        label_names = (
            [k for k, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
            if label_to_idx else None
        )

        logger.info(
            "TabularClassicalExtractor: %d samples, %d features.",
            X.shape[0], X.shape[1],
        )
        return FeatureSet(
            features=X,
            feature_type=self.feature_type,
            modality=self.modality,
            metadata=all_meta,
            labels=labels,
            label_names=label_names,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self._numerical_cols_arg is not None and self._categorical_cols_arg is not None:
            return self._numerical_cols_arg, self._categorical_cols_arg

        num_cols: list[str] = []
        cat_cols: list[str] = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                num_cols.append(col)
            else:
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= self.max_ohe_categories:
                    cat_cols.append(col)
                else:
                    logger.warning(
                        "Column %r has %d unique values (> max_ohe_categories=%d); "
                        "dropping from features.",
                        col, n_unique, self.max_ohe_categories,
                    )

        # Apply explicit overrides
        if self._numerical_cols_arg is not None:
            num_cols = [c for c in self._numerical_cols_arg if c in df.columns]
        if self._categorical_cols_arg is not None:
            cat_cols = [c for c in self._categorical_cols_arg if c in df.columns]

        return num_cols, cat_cols

    def _build_transformer(
        self,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> ColumnTransformer:
        ScalerClass = _SCALERS[self.scaler]
        transformers = []

        if num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=self.impute_numerical)),
                ("scaler",  ScalerClass()),
            ])
            transformers.append(("numerical", num_pipe, num_cols))

        if cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=self.impute_categorical,
                                          fill_value="missing")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                )),
            ])
            transformers.append(("categorical", cat_pipe, cat_cols))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",         # silently drop unrecognised columns
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _expand_datetimes(df: pd.DataFrame) -> pd.DataFrame:
        """Replace datetime columns with numerical component columns."""
        dt_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
            or (df[c].dtype == object and _looks_like_datetime(df[c]))
        ]
        for col in dt_cols:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                df = df.drop(columns=[col])
                df[f"{col}_year"]  = parsed.dt.year.astype("Int64")
                df[f"{col}_month"] = parsed.dt.month.astype("Int64")
                df[f"{col}_dow"]   = parsed.dt.dayofweek.astype("Int64")
                df[f"{col}_hour"]  = parsed.dt.hour.astype("Int64")
                df[f"{col}_min"]   = parsed.dt.minute.astype("Int64")
            except Exception:
                pass
        return df


def _looks_like_datetime(series: pd.Series, n_sample: int = 50) -> bool:
    """Heuristic: does this object column look like dates/times?"""
    sample = series.dropna().head(n_sample).astype(str)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8