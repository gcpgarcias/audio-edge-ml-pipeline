"""
Evaluation helpers shared by all Stage 3/4 trainers.

Functions
---------
compute_metrics     -- accuracy, F1, precision, recall, per-class breakdown,
                       confusion matrix, and (optional) ROC-AUC.
save_confusion_matrix_png
                    -- render a confusion matrix as a matplotlib heatmap.
log_run_to_mlflow   -- log params, scalar metrics, and artifact files to an
                       active MLflow run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    y_proba:     Optional[np.ndarray] = None,
    label_names: Optional[list[str]]  = None,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true:
        Integer ground-truth labels, shape ``(N,)``.
    y_pred:
        Integer predicted labels, shape ``(N,)``.
    y_proba:
        Soft probability estimates, shape ``(N, C)``.  Used for ROC-AUC;
        pass *None* to skip that metric.
    label_names:
        Ordered list of class name strings.  Used for per-class breakdown;
        if *None*, integer indices are used as keys.

    Returns
    -------
    dict
        Contains at minimum:

        ``val_accuracy``, ``val_f1_macro``, ``val_precision_macro``,
        ``val_recall_macro``, ``confusion_matrix`` (list-of-lists),
        ``per_class`` (dict of {class_name: {precision, recall, f1, support}}).

        Optionally: ``val_roc_auc_macro`` when *y_proba* is provided and
        there are ≥ 2 classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        classification_report,
    )

    n_classes = len(np.unique(y_true))
    names = label_names or [str(i) for i in range(n_classes)]

    metrics: dict = {
        "val_accuracy":        float(accuracy_score(y_true, y_pred)),
        "val_f1_macro":        float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "val_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "val_recall_macro":    float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "confusion_matrix":    confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class breakdown
    per_class: dict[str, dict] = {}
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=n_classes)
    for i, name in enumerate(names):
        if i < len(p_per):
            per_class[name] = {
                "precision": float(p_per[i]),
                "recall":    float(r_per[i]),
                "f1":        float(f_per[i]),
                "support":   int(support[i]),
            }
    metrics["per_class"] = per_class

    # ROC-AUC (only when probabilities provided and multi-class)
    if y_proba is not None and n_classes >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            multi = "ovr" if n_classes > 2 else "raise"
            auc = roc_auc_score(
                y_true, y_proba,
                multi_class=multi,
                average="macro",
            )
            metrics["val_roc_auc_macro"] = float(auc)
        except Exception as exc:
            logger.debug("ROC-AUC skipped: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix visualisation
# ---------------------------------------------------------------------------

def save_confusion_matrix_png(
    cm:          list | np.ndarray,
    label_names: list[str],
    path:        Path,
) -> None:
    """Render *cm* as a seaborn/matplotlib heatmap and save to *path*.

    Parameters
    ----------
    cm:
        Confusion matrix as a list-of-lists or numpy array.
    label_names:
        Class names for tick labels.
    path:
        Output PNG file path (parent directory must exist).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    n = len(label_names)
    fig_size = max(6, n)
    fig, ax = plt.subplots(figsize=(fig_size, max(5, n - 1)))

    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells with counts
    thresh = cm_arr.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=max(6, 10 - n // 5),
            )

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.debug("Confusion matrix saved: %s", path)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_run_to_mlflow(
    run,
    params:     dict,
    metrics:    dict,
    output_dir: Path,
    artifacts:  Optional[list[Path]] = None,
) -> None:
    """Log params, scalar metrics, and artifact files to *run*.

    Parameters
    ----------
    run:
        Active ``mlflow.ActiveRun`` context object (or *None* to skip).
    params:
        Flat dict of hyperparameters.  Values are coerced to strings.
    metrics:
        Dict of metric name → value.  Only ``int`` / ``float`` values are
        logged; nested dicts and lists (e.g. ``confusion_matrix``) are
        silently skipped.
    output_dir:
        Directory searched for standard artefact files when *artifacts* is
        *None*.  Checks for ``confusion_matrix.png``,
        ``classification_report.txt``, ``model_info.json``.
    artifacts:
        Explicit list of :class:`~pathlib.Path` objects to log.  When
        provided, overrides the auto-discovery from *output_dir*.
    """
    if run is None:
        return

    import mlflow

    # Params
    mlflow.log_params({k: str(v) for k, v in params.items()})

    # Scalar metrics only
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))

    # Artefact files
    if artifacts is not None:
        for art in artifacts:
            if art.exists():
                mlflow.log_artifact(str(art))
    else:
        for name in ("confusion_matrix.png",
                     "classification_report.txt",
                     "model_info.json"):
            art = output_dir / name
            if art.exists():
                mlflow.log_artifact(str(art))


# ---------------------------------------------------------------------------
# Text report helpers
# ---------------------------------------------------------------------------

def save_classification_report(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    label_names: list[str],
    path:        Path,
) -> None:
    """Write a sklearn classification_report to *path* as plain text."""
    from sklearn.metrics import classification_report
    try:
        report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
        path.write_text(report)
        logger.debug("Classification report saved: %s", path)
    except Exception as exc:
        logger.warning("Could not write classification report: %s", exc)


def save_model_info(
    output_dir:  Path,
    model_name:  str,
    run_name:    str,
    metrics:     dict,
    params:      dict,
    model_size_kb: float,
) -> None:
    """Write a ``model_info.json`` summary to *output_dir*."""
    info = {
        "model_name":          model_name,
        "run_name":            run_name,
        "model_size_kb":       model_size_kb,
        "params":              {k: str(v) for k, v in params.items()},
        "val_accuracy":        metrics.get("val_accuracy"),
        "val_f1_macro":        metrics.get("val_f1_macro"),
        "val_precision_macro": metrics.get("val_precision_macro"),
        "val_recall_macro":    metrics.get("val_recall_macro"),
        "val_roc_auc_macro":   metrics.get("val_roc_auc_macro"),
    }
    (output_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    logger.debug("model_info.json saved: %s", output_dir / "model_info.json")