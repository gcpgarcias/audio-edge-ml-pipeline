"""
Classical (sklearn) model trainers.

All seven trainers share a single ``SklearnTrainer`` implementation and are
registered under distinct names via ``@register_model``.

Registered names
----------------
``svm``            SVC with RBF kernel and probability calibration
``lda``            Linear Discriminant Analysis
``decision_tree``  Decision Tree classifier
``random_forest``  Random Forest classifier
``knn``            k-Nearest Neighbours classifier
``kmeans``         K-Means clustering (unsupervised — ignores labels during fit)
``pca_svm``        sklearn Pipeline: StandardScaler → PCA(50) → SVC

Input handling
--------------
All trainers flatten multi-dimensional features to 2-D ``(N, D)`` before
passing to the sklearn estimator (see :meth:`BaseTrainer.flatten`).

Persistence
-----------
Models are saved and loaded with :func:`joblib.dump` / :func:`joblib.load`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from .base import BaseTrainer, TrainResult
from . import register_model
from ..evaluate import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix_png,
    save_model_info,
    log_run_to_mlflow,
)

logger = logging.getLogger(__name__)


class SklearnTrainer(BaseTrainer):
    """Generic wrapper around a fitted sklearn estimator.

    Parameters
    ----------
    estimator:
        An unfitted sklearn estimator or Pipeline.
    **kwargs:
        Additional keyword arguments forwarded to the estimator constructor.
        Pass them as ``SklearnTrainer(estimator_factory, **params)`` — see
        each registered subclass below for the accepted keys.
    """

    model_type = "classical"

    def __init__(self, estimator, **kwargs):
        self._estimator = estimator
        self._fitted = False

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        label_names: list[str],
        run_name:    str,
        output_dir:  Path,
        mlflow_run,
    ) -> TrainResult:
        X_train = self.flatten(X_train)
        X_val   = self.flatten(X_val)

        is_kmeans = isinstance(self._estimator, KMeans)

        logger.info("Training %s on %d samples ...", self.name, len(X_train))
        if is_kmeans:
            self._estimator.fit(X_train)
            y_pred_val = self._estimator.predict(X_val)
            val_metrics: dict = {"note": "KMeans — cluster assignments, no supervised accuracy"}
        else:
            self._estimator.fit(X_train, y_train)
            y_pred_val  = self._estimator.predict(X_val)
            val_metrics = compute_metrics(y_val, y_pred_val, label_names=label_names)

        self._fitted = True

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{self.name}.joblib"
        self.save(model_path)
        model_size_kb = model_path.stat().st_size / 1024

        # Build params dict (needed by save_model_info and log_run_to_mlflow)
        params = {"model": self.name}
        if hasattr(self._estimator, "get_params"):
            params.update({k: str(v) for k, v in self._estimator.get_params().items()})

        # Write per-run artefacts via evaluate helpers
        if not is_kmeans:
            save_classification_report(y_val, y_pred_val, label_names,
                                       output_dir / "classification_report.txt")
            save_confusion_matrix_png(val_metrics.get("confusion_matrix", []),
                                      label_names, output_dir / "confusion_matrix.png")
        save_model_info(output_dir, self.name, run_name, val_metrics, params, model_size_kb)

        # Log to MLflow
        val_metrics["model_size_kb"] = model_size_kb
        log_run_to_mlflow(mlflow_run, params, val_metrics, output_dir)
        if mlflow_run is not None:
            import mlflow
            mlflow.log_artifact(str(model_path))

        return TrainResult(
            model_name=self.name,
            run_id=mlflow_run.info.run_id if mlflow_run else "",
            output_dir=output_dir,
            metrics=val_metrics,
            model_size_kb=model_size_kb,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(self.flatten(X))

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self._estimator, "predict_proba"):
            try:
                return self._estimator.predict_proba(self.flatten(X))
            except Exception:
                pass
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._estimator, path)
        logger.info("Model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "SklearnTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---------------------------------------------------------------------------
# Registered trainers — each is a thin factory function returning a
# SklearnTrainer pre-wired to the appropriate estimator.
# ---------------------------------------------------------------------------

def _make_trainer(name: str, factory):
    """Create a SklearnTrainer subclass registered under *name*."""

    cls = type(
        f"_{name.replace('_', ' ').title().replace(' ', '')}Trainer",
        (SklearnTrainer,),
        {"name": name, "model_type": "classical"},
    )

    @register_model
    class _Registered(cls):
        pass

    _Registered.__name__     = cls.__name__
    _Registered.__qualname__  = cls.__qualname__
    _Registered.name          = name

    # Override __init__ to accept **kwargs forwarded from config params
    original_init = _Registered.__init__

    def __init__(self, **kwargs):
        SklearnTrainer.__init__(self, factory(**kwargs))

    _Registered.__init__ = __init__
    return _Registered


# ---- SVM ---------------------------------------------------------------

@register_model
class SVMTrainer(SklearnTrainer):
    """Support Vector Machine with RBF kernel.

    Parameters
    ----------
    C:          Regularisation parameter (default 1.0).
    kernel:     Kernel type (default ``"rbf"``).
    gamma:      Kernel coefficient (default ``"scale"``).
    """

    name       = "svm"
    model_type = "classical"

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale", **_):
        super().__init__(
            SVC(C=C, kernel=kernel, gamma=gamma,
                probability=True, class_weight="balanced")
        )

    @classmethod
    def load(cls, path: Path) -> "SVMTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---- LDA ---------------------------------------------------------------

@register_model
class LDATrainer(SklearnTrainer):
    """Linear Discriminant Analysis.

    Parameters
    ----------
    n_components:   Number of components to keep (default *None* → min(n_classes-1, n_features)).
    solver:         LDA solver (default ``"svd"``).
    """

    name       = "lda"
    model_type = "classical"

    def __init__(self, n_components: Optional[int] = None, solver: str = "svd", **_):
        super().__init__(
            LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
        )

    @classmethod
    def load(cls, path: Path) -> "LDATrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---- Decision Tree -----------------------------------------------------

@register_model
class DecisionTreeTrainer(SklearnTrainer):
    """Decision Tree classifier.

    Parameters
    ----------
    max_depth:          Maximum tree depth (default *None* → unlimited).
    min_samples_leaf:   Minimum samples per leaf (default 1).
    """

    name       = "decision_tree"
    model_type = "classical"

    def __init__(self, max_depth: Optional[int] = None, min_samples_leaf: int = 1, **_):
        super().__init__(
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
            )
        )

    @classmethod
    def load(cls, path: Path) -> "DecisionTreeTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---- Random Forest -----------------------------------------------------

@register_model
class RandomForestTrainer(SklearnTrainer):
    """Random Forest classifier.

    Parameters
    ----------
    n_estimators:   Number of trees (default 100).
    max_depth:      Maximum depth per tree (default *None*).
    """

    name       = "random_forest"
    model_type = "classical"

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **_):
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                n_jobs=-1,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RandomForestTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---- K-NN --------------------------------------------------------------

@register_model
class KNNTrainer(SklearnTrainer):
    """k-Nearest Neighbours classifier.

    Parameters
    ----------
    n_neighbors:    Number of neighbours (default 5).
    metric:         Distance metric (default ``"minkowski"``).
    """

    name       = "knn"
    model_type = "classical"

    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", **_):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric))

    @classmethod
    def load(cls, path: Path) -> "KNNTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst


# ---- K-Means (unsupervised) --------------------------------------------

@register_model
class KMeansTrainer(SklearnTrainer):
    """K-Means clustering (unsupervised).

    Labels are ignored during training.  ``predict()`` returns cluster
    indices, not class labels.

    Parameters
    ----------
    n_clusters:     Number of clusters (default matches number of classes at
                    fit time; override if needed).
    n_init:         Number of K-Means initialisations (default 10).
    """

    name       = "kmeans"
    model_type = "classical"

    def __init__(self, n_clusters: Optional[int] = None, n_init: int = 10, **_):
        self._n_clusters_override = n_clusters
        self._n_init              = n_init
        # Estimator is created lazily at fit time so we know n_classes
        super().__init__(None)

    def fit(self, X_train, y_train, X_val, y_val, label_names,
            run_name, output_dir, mlflow_run) -> TrainResult:
        n_clusters = self._n_clusters_override or len(label_names)
        self._estimator = KMeans(n_clusters=n_clusters, n_init=self._n_init, random_state=42)
        return super().fit(X_train, y_train, X_val, y_val, label_names,
                           run_name, output_dir, mlflow_run)

    @classmethod
    def load(cls, path: Path) -> "KMeansTrainer":
        inst = cls.__new__(cls)
        inst._estimator             = joblib.load(path)
        inst._fitted                = True
        inst._n_clusters_override   = None
        inst._n_init                = 10
        return inst


# ---- PCA + SVM pipeline ------------------------------------------------

@register_model
class PCASVMTrainer(SklearnTrainer):
    """Pipeline: StandardScaler → PCA → SVM.

    Reduces feature dimensionality via PCA before fitting an RBF SVM.
    Effective for high-dimensional classical or deep embedding features.

    Parameters
    ----------
    n_components:   PCA components to keep (default 50).
    C:              SVM regularisation parameter (default 1.0).
    kernel:         SVM kernel (default ``"rbf"``).
    """

    name       = "pca_svm"
    model_type = "classical"

    def __init__(self, n_components: int = 50, C: float = 1.0, kernel: str = "rbf", **_):
        super().__init__(
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n_components)),
                ("svm",    SVC(C=C, kernel=kernel, probability=True, class_weight="balanced")),
            ])
        )

    @classmethod
    def load(cls, path: Path) -> "PCASVMTrainer":
        inst = cls.__new__(cls)
        inst._estimator = joblib.load(path)
        inst._fitted    = True
        return inst

