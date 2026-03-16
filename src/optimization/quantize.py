"""
Stage 6b utilities — ONNX conversion and multi-mode quantization.

All functions are pure utilities with no CLI or side effects.

Functions
---------
detect_model_type       — infer "sklearn" or "keras" from artifact directory
find_model_file         — resolve path to .joblib or .keras file
convert_to_onnx         — sklearn/Keras → ONNX fp32 baseline
optimize_dynamic_int8   — ONNX dynamic INT8 (weights only)
optimize_static_int8    — ONNX static INT8 (calibrated activations)
optimize_float16        — ONNX float16 weight conversion
evaluate_onnx           — ONNX inference benchmark → {accuracy, latency_ms}
convert_to_tflite_fp32    — Keras → TFLite fp32 baseline
convert_to_tflite_dynamic — Keras → TFLite dynamic range INT8
convert_to_tflite_int8    — Keras → TFLite full integer INT8 (int8 I/O)
convert_to_tflite_float16 — Keras → TFLite float16
evaluate_tflite           — TFLite inference benchmark → {accuracy, latency_ms}

Dependencies
------------
    pip install onnx skl2onnx onnxruntime onnxconverter-common
    pip install tf2onnx tensorflow   # only needed for Keras models
    # tensorflow includes tf.lite — no extra package needed for TFLite
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
        # Run tf2onnx in a subprocess to avoid Metal GPU hang on Apple Silicon.
        #
        # Root cause: importing `src.preprocessing.pipeline` (done at optimize.py
        # module level to load FeatureSet data) pulls in all deep feature
        # extractors, which import TensorFlow.  By the time convert_to_onnx()
        # is called, Metal PluggableDevice is already initialised and
        # tf.config.set_visible_devices() is too late.  tf2onnx then creates a
        # TF v1 session that hangs on the Metal device.
        #
        # A fresh subprocess has no prior TF import, so set_visible_devices()
        # takes effect before any GPU initialisation occurs.
        import subprocess
        import sys

        # Build the script as a single importable block so Python can handle
        # quoting/escaping uniformly.
        script = "\n".join([
            "import os",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
            "import tensorflow as tf",
            "tf.config.set_visible_devices([], 'GPU')",
            f"model = tf.keras.models.load_model({str(model_path)!r})",
            "import onnx, tf2onnx.convert",
            "spec = [tf.TensorSpec(",
            "    shape=[None] + list(model.input_shape[1:]),",
            "    dtype=tf.float32,",
            "    name='input',",
            ")]",
            "onnx_model, _ = tf2onnx.convert.from_keras(",
            "    model, input_signature=spec, opset=13",
            ")",
            f"onnx.save(onnx_model, {str(output_path)!r})",
        ])
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tf2onnx conversion subprocess failed (exit {result.returncode}):\n"
                + result.stderr[-3000:]
            )

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

    # Infer the actual input name and expected shape from the ONNX model so
    # the reader works for both sklearn (input='float_input', rank 2) and
    # Keras / CNN (input='input', rank 3-4) models.
    try:
        import onnxruntime as _ort
    except ImportError:
        raise ImportError("pip install onnxruntime")
    _info = _ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _input_name = _info.get_inputs()[0].name
    _onnx_shape = _info.get_inputs()[0].shape  # [batch, d1, ...]
    _spatial: list = []
    if len(_onnx_shape) > 2:
        for d in _onnx_shape[1:]:
            if isinstance(d, int) and d > 0:
                _spatial.append(d)
            else:
                _spatial = []
                break
        if _spatial and int(np.prod(_spatial)) != calib.shape[1]:
            _spatial = []

    class _Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray) -> None:
            self._data = data
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self._data):
                return None
            sample = self._data[self._idx : self._idx + 1]
            if _spatial:
                sample = sample.reshape([1] + _spatial)
            self._idx += 1
            return {_input_name: sample}

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
    model_fp16 = onnx_fp16.convert_float_to_float16(model, keep_io_types=True)
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

    # X_f arrives flattened (N, n_features_flat) from optimize.py.
    # For CNN/RNN models the ONNX graph expects a higher-rank input
    # (e.g. (N, H, W, C)); restore the shape from the ONNX metadata.
    onnx_in_shape = sess.get_inputs()[0].shape  # [batch_or_None, d1, ...]
    if len(onnx_in_shape) > 2:
        spatial = []
        for d in onnx_in_shape[1:]:
            if isinstance(d, int) and d > 0:
                spatial.append(d)
            else:
                spatial = []
                break
        if spatial and int(np.prod(spatial)) == X_f.shape[1]:
            X_f = X_f.reshape([N] + spatial)

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


# ---------------------------------------------------------------------------
# TFLite conversion (Keras models only)
# ---------------------------------------------------------------------------
#
# All TFLite conversion functions run in a subprocess to be consistent with
# the ONNX Keras path.  TFLite conversion is a compiler step (MLIR) — it never
# executes inference on Metal — so the Metal PluggableDevice does not cause a
# hang.  set_visible_devices() is intentionally NOT called in these subprocesses:
# on Apple Silicon, importing tensorflow registers the Metal PluggableDevice at
# plugin-load time, so calling set_visible_devices() after that raises
# RuntimeError and causes every TFLite subprocess to exit non-zero.

def _run_subprocess_script(script: str, timeout: int = 300) -> None:
    """Execute *script* in a subprocess; raise RuntimeError on failure."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {result.returncode}):\n"
            + result.stderr[-3000:]
        )


def convert_to_tflite_fp32(keras_model_path: Path, output_path: Path) -> Path:
    """Convert a Keras model to TFLite fp32 (no quantization)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Note: set_visible_devices is intentionally omitted here.
    # TFLite conversion is a compiler step — it never executes on Metal —
    # so Metal initialisation does not cause a hang.  Calling
    # set_visible_devices() after `import tensorflow` raises RuntimeError
    # because the Metal PluggableDevice is already registered at plugin-load
    # time, which would silently break every TFLite mode.
    _run_subprocess_script("\n".join([
        "import os",
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
        "import tensorflow as tf",
        f"model = tf.keras.models.load_model({str(keras_model_path)!r})",
        "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
        "tflite_model = converter.convert()",
        f"open({str(output_path)!r}, 'wb').write(tflite_model)",
    ]))
    logger.info("TFLite fp32    → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def convert_to_tflite_dynamic(keras_model_path: Path, output_path: Path) -> Path:
    """Convert a Keras model to TFLite with dynamic range INT8 quantization.

    Weights are quantized to INT8; activations are quantized dynamically at
    runtime.  No calibration data required.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_subprocess_script("\n".join([
        "import os",
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
        "import tensorflow as tf",
        f"model = tf.keras.models.load_model({str(keras_model_path)!r})",
        "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
        "converter.optimizations = [tf.lite.Optimize.DEFAULT]",
        "tflite_model = converter.convert()",
        f"open({str(output_path)!r}, 'wb').write(tflite_model)",
    ]))
    logger.info("TFLite dynamic → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def convert_to_tflite_int8(
    keras_model_path: Path,
    output_path: Path,
    X_calib: np.ndarray,
) -> Path:
    """Convert a Keras model to fully integer TFLite (INT8 weights, activations,
    and I/O).

    Uses up to 100 calibration samples to compute activation ranges.  The
    resulting model has ``int8`` input and output tensors — optimal for
    TFLite Micro on Nicla Vision.
    """
    import os
    import tempfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise calibration data to a temp file so the subprocess can load it.
    calib_fd, calib_path = tempfile.mkstemp(suffix=".npy")
    os.close(calib_fd)
    try:
        np.save(calib_path, X_calib[:100].astype(np.float32))
        _run_subprocess_script("\n".join([
            "import os, numpy as np",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
            "import tensorflow as tf",
            f"model = tf.keras.models.load_model({str(keras_model_path)!r})",
            f"calib_data = np.load({calib_path!r})",
            "input_shape = tuple(model.input_shape[1:])",
            "def representative_data_gen():",
            "    for sample in calib_data:",
            "        yield [sample.reshape((1,) + input_shape)]",
            "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
            "converter.optimizations = [tf.lite.Optimize.DEFAULT]",
            "converter.representative_dataset = representative_data_gen",
            "converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]",
            "converter.inference_input_type  = tf.int8",
            "converter.inference_output_type = tf.int8",
            "tflite_model = converter.convert()",
            f"open({str(output_path)!r}, 'wb').write(tflite_model)",
        ]))
    finally:
        os.unlink(calib_path)
    logger.info("TFLite INT8    → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def convert_to_tflite_float16(keras_model_path: Path, output_path: Path) -> Path:
    """Convert a Keras model to TFLite with float16 weight quantization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_subprocess_script("\n".join([
        "import os",
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",
        "import tensorflow as tf",
        f"model = tf.keras.models.load_model({str(keras_model_path)!r})",
        "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
        "converter.optimizations = [tf.lite.Optimize.DEFAULT]",
        "converter.target_spec.supported_types = [tf.float16]",
        "tflite_model = converter.convert()",
        f"open({str(output_path)!r}, 'wb').write(tflite_model)",
    ]))
    logger.info("TFLite float16 → %s  (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


# ---------------------------------------------------------------------------
# TFLite evaluation
# ---------------------------------------------------------------------------

def evaluate_tflite(
    tflite_path: Path,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Benchmark a TFLite model; return ``{accuracy, latency_ms}``.

    Parameters
    ----------
    tflite_path:
        Path to the ``.tflite`` file.
    X:
        Feature matrix, shape ``(N, *feature_dims)`` or ``(N, n_features_flat)``.
        Reshaped per-sample to match the model's expected input shape.
    y:
        Integer class indices, shape ``(N,)``.

    Notes
    -----
    Inference is run sample-by-sample to match edge-device behaviour where
    batching is not available.  For fully-integer INT8 models, input samples
    are quantized and outputs are dequantized using the stored scale / zero-point.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_idx  = input_details[0]["index"]
    output_idx = output_details[0]["index"]

    # Resize interpreter to single-sample batches
    model_input_shape = list(input_details[0]["shape"])  # e.g. [1, 128, 13]
    single_shape      = [1] + model_input_shape[1:]
    interpreter.resize_tensor_input(input_idx, single_shape)
    interpreter.allocate_tensors()

    input_dtype  = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]
    n_elements   = int(np.prod(model_input_shape[1:]))

    # Quantization params (used only for INT8 I/O models)
    in_scale, in_zp   = input_details[0].get("quantization", (1.0, 0))
    out_scale, out_zp = output_details[0].get("quantization", (1.0, 0))

    y_pred = []
    t0 = time.perf_counter()

    for sample in X:
        inp = sample.flat[:n_elements].reshape(single_shape).astype(np.float32)
        if input_dtype == np.int8:
            inp = np.round(inp / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_idx, inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_idx)[0]  # shape (n_classes,)
        if output_dtype == np.int8:
            out = (out.astype(np.float32) - out_zp) * out_scale
        y_pred.append(int(np.argmax(out)))

    elapsed    = time.perf_counter() - t0
    latency_ms = elapsed / len(X) * 1000.0
    accuracy   = float(np.mean(np.array(y_pred) == y.astype(np.int64)))
    return {"accuracy": accuracy, "latency_ms": latency_ms}
