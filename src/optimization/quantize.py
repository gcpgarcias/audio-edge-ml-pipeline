"""
Stage 6b utilities — ONNX conversion and multi-mode quantization.

All functions are pure utilities with no CLI or side effects.

Functions
---------
detect_model_type   — infer "sklearn" or "keras" from artifact directory
find_model_file     — resolve path to .joblib or .keras file
convert_to_onnx     — sklearn/Keras → ONNX fp32 baseline
optimize_dynamic_int8
optimize_static_int8
optimize_float16
evaluate_onnx       — run inference, return {accuracy, latency_ms}

Dependencies
------------
    pip install onnx skl2onnx onnxruntime onnxconverter-common
    pip install tf2onnx tensorflow   # only needed for Keras models
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def detect_model_type(
    artifact_uri: str,
    model_name: str,
) -> Literal["sklearn", "keras"]:
    """Return ``'sklearn'`` or ``'keras'`` based on files present in *artifact_uri*.

    Raises ``FileNotFoundError`` if neither ``.joblib`` nor ``.keras`` is found.
    """
    base = Path(artifact_uri)
    if (base / f"{model_name}.joblib").exists():
        return "sklearn"
    for suffix in (f"{model_name}.keras", "model.keras"):
        if (base / suffix).exists():
            return "keras"
    raise FileNotFoundError(
        f"No model file found for '{model_name}' in {artifact_uri}. "
        f"Expected {model_name}.joblib or *.keras"
    )


def find_model_file(artifact_uri: str, model_name: str) -> Path:
    """Return the resolved path to the model file inside *artifact_uri*."""
    base = Path(artifact_uri)
    candidates = [
        base / f"{model_name}.joblib",
        base / f"{model_name}.keras",
        base / "model.keras",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No model file found for '{model_name}' in {artifact_uri}"
    )


# ---------------------------------------------------------------------------
# ONNX conversion
# ---------------------------------------------------------------------------

def convert_to_onnx(
    model_path: Path,
    n_features: int,
    output_path: Path,
) -> Path:
    """Convert a sklearn ``.joblib`` or Keras model to ONNX (fp32 baseline).

    Parameters
    ----------
    model_path:
        Path to ``.joblib`` (sklearn) or ``.keras`` / ``.h5`` (Keras).
    n_features:
        Number of input features (after flattening).  Used only for sklearn.
    output_path:
        Destination ``.onnx`` file path.
    """
    try:
        import onnx  # noqa: F401 — verify onnx is installed early
    except ImportError:
        raise ImportError("pip install onnx")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    if model_path.suffix == ".joblib":
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("pip install skl2onnx")

        model = joblib.load(model_path)
        initial_types = [("float_input", FloatTensorType([None, n_features]))]
        options = {type(model): {"zipmap": False}}
        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                options=options,
            )
        except Exception:
            # Some Pipeline wrappers reject per-type options — retry without
            onnx_model = convert_sklearn(model, initial_types=initial_types)

        output_path.write_bytes(onnx_model.SerializeToString())

    elif model_path.suffix in (".keras", ".h5"):
        try:
            import tensorflow as tf
            import tf2onnx.convert
        except ImportError:
            raise ImportError("pip install tf2onnx tensorflow")

        model = tf.keras.models.load_model(model_path)
        import onnx as _onnx
        onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
        _onnx.save(onnx_model, str(output_path))

    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}")

    size_kb = output_path.stat().st_size / 1024
    logger.info("ONNX fp32  → %s  (%.1f KB)", output_path.name, size_kb)
    return output_path


# ---------------------------------------------------------------------------
# Optimization modes
# ---------------------------------------------------------------------------

def optimize_dynamic_int8(onnx_path: Path, output_path: Path) -> Path:
    """Dynamic INT8 quantization — weights only, no calibration data needed."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("pip install onnxruntime")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("Dynamic INT8 → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def optimize_static_int8(
    onnx_path: Path,
    output_path: Path,
    X_calib: np.ndarray,
) -> Path:
    """Static INT8 quantization — calibrates activations with up to 50 samples."""
    try:
        from onnxruntime.quantization import (
            quantize_static,
            QuantType,
            CalibrationDataReader,
        )
    except ImportError:
        raise ImportError("pip install onnxruntime")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calib = X_calib[:50].astype(np.float32)

    class _Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray) -> None:
            self._data = data
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self._data):
                return None
            row = {"float_input": self._data[self._idx : self._idx + 1]}
            self._idx += 1
            return row

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=_Reader(calib),
        weight_type=QuantType.QInt8,
    )
    logger.info("Static INT8  → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def optimize_float16(onnx_path: Path, output_path: Path) -> Path:
    """Float16 weight conversion."""
    try:
        import onnx
        from onnxconverter_common import float16 as onnx_fp16
    except ImportError:
        raise ImportError("pip install onnx onnxconverter-common")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    model_fp16 = onnx_fp16.convert_float_to_float16(model)
    onnx.save(model_fp16, str(output_path))
    logger.info("Float16      → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_onnx(
    onnx_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    label_names: Optional[list] = None,
) -> dict:
    """Run inference with onnxruntime; return ``{accuracy, latency_ms}``.

    Parameters
    ----------
    onnx_path:
        Path to the ``.onnx`` model file.
    X:
        Float32 feature matrix, shape ``(N, n_features)``.
    y:
        Integer class indices, shape ``(N,)``.
    label_names:
        Required only when the ONNX model outputs string class labels.

    Returns
    -------
    dict with keys ``accuracy`` (float) and ``latency_ms`` (mean per-sample ms).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    X_f = X.astype(np.float32)
    N = len(X_f)

    # Warm-up pass
    sess.run(None, {input_name: X_f[:1]})

    t0 = time.perf_counter()
    outputs = sess.run(None, {input_name: X_f})
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed / N * 1000.0

    first_out = outputs[0]
    if first_out.ndim == 2:
        # Keras-style: (N, n_classes) → argmax
        y_pred = np.argmax(first_out, axis=1).astype(np.int64)
    else:
        # sklearn-style: (N,) class labels (int64 or string)
        if first_out.dtype.kind in ("U", "S", "O"):
            if label_names is None:
                raise ValueError(
                    "label_names required to decode string ONNX output labels"
                )
            name_to_idx = {n: i for i, n in enumerate(label_names)}
            y_pred = np.array(
                [name_to_idx.get(str(lbl), -1) for lbl in first_out],
                dtype=np.int64,
            )
        else:
            y_pred = first_out.astype(np.int64)

    accuracy = float(np.mean(y_pred == y.astype(np.int64)))
    return {"accuracy": accuracy, "latency_ms": latency_ms}
