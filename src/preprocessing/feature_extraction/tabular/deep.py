"""
Deep tabular feature extractor.

Extends the classical preprocessing pipeline with degree-2 polynomial
expansion to capture pairwise interaction terms between numerical features.
This is the standard way to give linear models (SVM, Logistic Regression)
the ability to learn non-linear decision boundaries without switching to a
tree-based model, and it also provides richer inputs to neural networks
trained on tabular data (TabNet, SAINT, FT-Transformer).

Extractor                  name                Output shape   Suited for
-------------------------  ------------------  -------------  ----------------------
TabularPolynomialExtractor tabular_polynomial  (D_poly,)      SVM, linear models,
                                                              shallow NNs on tabular

D_poly = C(n_features + degree, degree) for ``interaction_only=False``.
With 20 numerical features and degree=2: (20 + 1)(20 + 2)/2 = 231 features.

Design note
-----------
``TabularPolynomialExtractor`` inherits the full preprocessing pipeline from
:class:`TabularClassicalExtractor` (imputation + scaling + OHE) and adds a
``PolynomialFeatures`` step applied only to the numerical portion.
Categorical OHE columns are concatenated as-is (polynomial expansion on
binary indicator columns is redundant and would explode the feature space).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ..base import BaseDatasetLoader, BaseFeatureExtractor, FeatureSet
from ..registry import register
from .classical import (
    TabularClassicalExtractor,
    _SCALERS,
)

logger = logging.getLogger(__name__)


@register
class TabularPolynomialExtractor(BaseFeatureExtractor):
    """Tabular features with degree-2 polynomial expansion of numerical columns.

    The pipeline is:
        numerical columns  → impute → scale → PolynomialFeatures(degree)
        categorical columns → impute → OneHotEncoder

    Both parts are concatenated into a single flat vector.

    Parameters
    ----------
    degree:
        Polynomial degree.  2 (default) captures all pairwise interactions.
        Use 1 for the same result as ``TabularClassicalExtractor`` without
        the overhead of this class.
    interaction_only:
        If *True*, only cross-product terms are produced (no ``x_i^2``).
        Reduces output dimensionality and avoids redundancy with the scaled
        input features.
    include_bias:
        Whether to include a bias (constant 1) term.
    numerical_cols, categorical_cols, label_col, scaler,
    impute_numerical, impute_categorical, max_ohe_categories:
        Same as :class:`TabularClassicalExtractor`.
    """

    name         = "tabular_polynomial"
    feature_type = "deep"
    modality     = "tabular"

    def __init__(
        self,
        degree:               int                 = 2,
        interaction_only:     bool                = False,
        include_bias:         bool                = False,
        numerical_cols:       Optional[list[str]] = None,
        categorical_cols:     Optional[list[str]] = None,
        label_col:            Optional[str]       = None,
        scaler:               str                 = "standard",
        impute_numerical:     str                 = "median",
        impute_categorical:   str                 = "most_frequent",
        max_ohe_categories:   int                 = 50,
    ) -> None:
        self.degree           = degree
        self.interaction_only = interaction_only
        self.include_bias     = include_bias

        # Reuse the classical extractor for shared preprocessing
        self._classical = TabularClassicalExtractor(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            label_col=label_col,
            scaler=scaler,
            impute_numerical=impute_numerical,
            impute_categorical=impute_categorical,
            max_ohe_categories=max_ohe_categories,
        )
        self._poly:     Optional[PolynomialFeatures] = None
        self._n_num:    int = 0  # number of numerical features after scaling

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        """Transform a single row using the fitted pipeline."""
        if self._poly is None:
            raise RuntimeError(
                "Transformer not fitted.  Call extract_dataset() first."
            )
        classical_vec = self._classical.extract(sample_path, **kwargs)
        return self._apply_poly(classical_vec[np.newaxis, :])[0]

    def extract_dataset(
        self,
        loader: BaseDatasetLoader,
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        """Fit preprocessing + polynomial expansion, then transform."""
        # Step 1: classical fit+transform (impute + scale + OHE)
        classical_fs = self._classical.extract_dataset(loader, max_samples)
        X_classical  = classical_fs.features  # (N, D_classical)

        # Step 2: identify how many numerical features the classical transformer
        #         produced (they come first in ColumnTransformer output).
        ct = self._classical._transformer
        if ct is None:
            raise RuntimeError("Classical transformer was not fitted.")

        n_num = 0
        for name, pipe, cols in ct.transformers_:
            if name == "numerical":
                n_num = len(cols)  # after scaling, same dimensionality
                break
        self._n_num = n_num

        # Step 3: fit PolynomialFeatures on numerical columns only
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        X_num_poly = self._poly.fit_transform(
            X_classical[:, :n_num]
        )                                            # (N, D_poly)
        X_cat      = X_classical[:, n_num:]          # (N, D_cat)
        X_combined = np.concatenate([X_num_poly, X_cat], axis=1)

        logger.info(
            "TabularPolynomialExtractor: %d samples, %d classical → %d poly+cat features.",
            X_combined.shape[0], X_classical.shape[1], X_combined.shape[1],
        )

        return FeatureSet(
            features=X_combined.astype(np.float32),
            feature_type=self.feature_type,
            modality=self.modality,
            metadata=classical_fs.metadata,
            labels=classical_fs.labels,
            label_names=classical_fs.label_names,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_poly(self, X: np.ndarray) -> np.ndarray:
        X_num_poly = self._poly.transform(X[:, :self._n_num])
        X_cat      = X[:, self._n_num:]
        return np.concatenate([X_num_poly, X_cat], axis=1).astype(np.float32)