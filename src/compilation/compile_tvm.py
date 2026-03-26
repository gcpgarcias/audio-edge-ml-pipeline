"""
Stage 6 — TVM model compilation.

Converts a Stage-5 ONNX model to a TVM compiled library, benchmarks
latency against the ONNX Runtime baseline, and optionally auto-tunes
operator schedules with MetaSchedule.

Two compilation levels are available:
  baseline  — relay.build() at opt_level 3 (seconds)
  tuned     — MetaSchedule auto-tuning (minutes–hours, much faster inference)

Usage
-----
    # Baseline compilation
    python -m src.compilation.compile_tvm \\
        --model        data/models/optimized/.../model_fp32.onnx \\
        --features-val data/processed/fsc22_melspec_val \\
        --output       data/models/compiled/fsc22_tvm

    # With MetaSchedule tuning
    python -m src.compilation.compile_tvm \\
        --model        data/models/optimized/.../model_fp32.onnx \\
        --features-val data/processed/fsc22_melspec_val \\
        --output       data/models/compiled/fsc22_tvm \\
        --tune --n-trials 400

    # Via YAML
    python -m src.compilation.compile_tvm --config config/compilation_tvm.yaml

Config schema (compilation_tvm.yaml)
--------------------------------------
::

    model:       data/models/optimized/.../model_fp32.onnx
    features:    data/processed/fsc22_melspec_val
    output:      data/models/compiled/fsc22_tvm
    class_filter: [Axe, BirdChirping, ...]   # optional
    target:      llvm                         # default; or llvm -mcpu=native
    opt_level:   3
    tune:        false
    n_trials:    400
    experiment:  fsc22-tvm-compilation

Output layout (output_dir/)
----------------------------
    model_baseline.so       compiled shared library (no tuning)
    model_tuned.so          compiled shared library (after tuning, if --tune)
    tuning_logs/            MetaSchedule database (if --tune)
    tvm_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_onnx_shape(model_path: Path) -> tuple[str, list[int]]:
    """Return (input_name, static_shape) with batch dim fixed to 1."""
    import onnx
    m = onnx.load(str(model_path))
    inp  = m.graph.input[0]
    name = inp.name or "input"
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    dims[0] = 1   # fix dynamic batch to 1
    return name, dims


def _apply_class_filter(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list,
    class_filter: Optional[list],
) -> tuple[np.ndarray, np.ndarray, list]:
    """Filter to class_filter classes; sort alphabetically for a canonical encoding."""
    if not class_filter:
        return X, y, label_names
    filter_set    = set(class_filter)
    allowed_pairs = sorted(
        [(i, n) for i, n in enumerate(label_names) if n in filter_set],
        key=lambda p: p[1],
    )
    allowed_idx = [i for i, _ in allowed_pairs]
    if not allowed_idx:
        raise ValueError(f"class_filter matched no classes in {label_names}")
    mask    = np.isin(y, allowed_idx)
    X, y    = X[mask], y[mask]
    idx_map = {old: new for new, old in enumerate(allowed_idx)}
    y       = np.array([idx_map[lbl] for lbl in y], dtype=y.dtype)
    label_names = [p[1] for p in allowed_pairs]
    logger.info("class_filter — %d classes, %d samples", len(label_names), len(X))
    return X, y, label_names


def _onnxrt_latency(model_path: Path, sample: np.ndarray, input_name: str,
                    n_runs: int = 50) -> float:
    """Return mean per-sample latency (ms) using ONNX Runtime as baseline."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])
    inp  = {input_name: sample}
    # Warm-up
    for _ in range(5):
        sess.run(None, inp)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, inp)
    return (time.perf_counter() - t0) / n_runs * 1000


# ---------------------------------------------------------------------------
# Relax helpers  (TVM 0.16+ — Relay has been replaced by Relax)
# ---------------------------------------------------------------------------

def _build_relax(onnx_path: Path, input_name: str, input_shape: list,
                 target, opt_level: int = 3):
    """Load ONNX → Relax IRModule → compiled Executable."""
    import onnx
    import tvm
    import tvm.relax as relax
    from tvm.relax.frontend.onnx import from_onnx

    logger.info("Converting ONNX → Relax …")
    onnx_model = onnx.load(str(onnx_path))
    mod = from_onnx(
        onnx_model,
        shape_dict={input_name: input_shape},
        dtype_dict="float32",
        sanitize_input_names=False,
    )

    mod = relax.transform.LegalizeOps()(mod)

    logger.info("Compiling (opt_level=%d, target=%s) …", opt_level, target)
    with tvm.transform.PassContext(opt_level=opt_level):
        ex = relax.build(mod, target=target)

    return ex


def _to_tvm_tensor(arr: np.ndarray, dev):
    """Convert a numpy array to a TVM runtime Tensor on the given device."""
    import tvm
    arr = np.ascontiguousarray(arr.astype(np.float32))
    t   = tvm.runtime.empty(arr.shape, dtype="float32", device=dev)
    t.copyfrom(arr)
    return t


def _tvm_latency(ex, dev, input_name: str, sample: np.ndarray,
                 n_runs: int = 50) -> float:
    """Return mean per-sample latency (ms) using TVM Relax VirtualMachine."""
    import tvm.relax as relax

    vm  = relax.VirtualMachine(ex, dev)
    inp = _to_tvm_tensor(sample, dev)

    # Warm-up
    for _ in range(5):
        vm["main"](inp)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        vm["main"](inp)
    return (time.perf_counter() - t0) / n_runs * 1000


def _tvm_accuracy(ex, dev, input_name: str,
                  X: np.ndarray, y: np.ndarray,
                  input_shape: list) -> float:
    """Run full validation set through TVM Relax VM; return accuracy."""
    import tvm.relax as relax

    vm      = relax.VirtualMachine(ex, dev)
    correct = 0
    for sample, label in zip(X, y):
        inp  = _to_tvm_tensor(sample.reshape(input_shape), dev)
        out  = vm["main"](inp)
        pred = int(np.argmax(out.numpy()))
        correct += int(pred == label)
    return correct / len(y)


# ---------------------------------------------------------------------------
# MetaSchedule tuning (optional)
# ---------------------------------------------------------------------------

def _tune_with_metaschedule(onnx_path: Path, input_name: str,
                             input_shape: list, target,
                             work_dir: Path, n_trials: int):
    """Auto-tune with MetaSchedule (Relax); return tuned Executable."""
    import onnx
    import tvm.relax as relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import meta_schedule as ms

    logger.info("MetaSchedule tuning — %d trials, work_dir=%s", n_trials, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.load(str(onnx_path))
    mod = from_onnx(
        onnx_model,
        shape_dict={input_name: input_shape},
        dtype_dict="float32",
        sanitize_input_names=False,
    )
    # Legalize high-level Relax ops → TIR primitives so MetaSchedule can
    # extract and schedule individual operator kernels.
    mod = relax.transform.LegalizeOps()(mod)

    # Run generated kernels single-threaded inside the evaluation subprocess.
    # Multi-threaded benchmark workers crash on Apple Silicon when a schedule
    # generates a parallelised kernel with bad vectorisation or alignment.
    def _single_thread_init():
        import os
        os.environ["TVM_NUM_THREADS"] = "1"

    from tvm.s_tir.meta_schedule.runner import LocalRunner, EvaluatorConfig
    runner = LocalRunner(
        timeout_sec      = 60,
        evaluator_config = EvaluatorConfig(number=3, repeat=1, min_repeat_ms=0),
        initializer      = _single_thread_init,
    )

    with ms.Profiler() as profiler:
        database = ms.relax_integration.tune_relax(
            mod               = mod,
            params            = {},
            target            = target,
            work_dir          = str(work_dir),
            max_trials_global = n_trials,
            runner            = runner,
        )

    logger.info("Tuning profile:\n%s", profiler.table())

    logger.info("Compiling with tuned schedules …")
    ex_tuned = ms.relax_integration.compile_relax(
        database = database,
        mod      = mod,
        params   = {},
        target   = target,
    )

    return ex_tuned


# ---------------------------------------------------------------------------
# Main compilation entry point
# ---------------------------------------------------------------------------

def compile_model(
    model_path:   Path,
    features_dir: Path,
    output_dir:   Path,
    class_filter: Optional[list],
    target_str:   str   = "llvm",
    opt_level:    int   = 3,
    tune:         bool  = False,
    n_trials:     int   = 400,
    experiment:   str   = "tvm-compilation",
    mlflow_uri:   Optional[str] = None,
) -> dict:
    import tvm

    import subprocess, platform
    output_dir.mkdir(parents=True, exist_ok=True)

    # On Apple Silicon "llvm" alone generates scalar code.
    # apple-latest enables SME (M4+) which causes SIGILL on M2 — use apple-m2 explicitly.
    if target_str == "llvm" and platform.system() == "Darwin" and platform.machine() == "arm64":
        target_str = '{"kind": "llvm", "mtriple": "arm64-apple-macosx", "mcpu": "apple-m2"}'

    target = tvm.target.Target(target_str)

    # MetaSchedule requires num-cores; inject it into the target when tuning
    if tune and target.attrs.get("num-cores") is None:
        if platform.system() == "Darwin":
            n_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
        else:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
        config = {k: target.attrs[k] for k in target.attrs.keys()}
        config["kind"] = target.kind.name
        config["num-cores"] = n_cores
        target = tvm.target.Target(config)
    dev    = tvm.device(target.kind.name, 0)

    # ── Validation features ──────────────────────────────────────────────────
    logger.info("Loading validation features from %s", features_dir)
    fs          = FeaturePipeline.load(features_dir)
    X, y        = fs.features, fs.labels
    label_names = fs.label_names or []
    if y is None:
        raise ValueError("Feature set has no labels.")
    X, y, label_names = _apply_class_filter(X, y, label_names, class_filter)
    logger.info("%d samples  %d classes  shape %s", len(X), len(label_names), X.shape[1:])

    # ── ONNX input metadata ───────────────────────────────────────────────────
    input_name, input_shape = _load_onnx_shape(model_path)
    logger.info("ONNX input: name=%r  shape=%s", input_name, input_shape)

    # ── ONNX Runtime baseline ─────────────────────────────────────────────────
    sample = X[0:1].astype(np.float32)
    # Reshape sample to match ONNX input shape if needed
    sample = sample.reshape(input_shape)

    logger.info("Benchmarking ONNX Runtime baseline …")
    onnx_latency = _onnxrt_latency(model_path, sample, input_name)
    logger.info("  ONNX Runtime latency: %.3f ms", onnx_latency)

    # ── Baseline TVM compilation ──────────────────────────────────────────────
    lib_baseline = _build_relax(model_path, input_name, input_shape,
                                target, opt_level)
    baseline_path = output_dir / "model_baseline.so"
    lib_baseline.export_library(str(baseline_path))
    logger.info("Baseline library → %s  (%.1f KB)",
                baseline_path.name, baseline_path.stat().st_size / 1024)

    tvm_latency_baseline = _tvm_latency(lib_baseline, dev, input_name, sample)
    logger.info("  TVM baseline latency: %.3f ms  (speedup %.2fx vs ONNX)",
                tvm_latency_baseline, onnx_latency / tvm_latency_baseline)

    logger.info("Evaluating baseline accuracy on %d samples …", len(X))
    tvm_accuracy_baseline = _tvm_accuracy(lib_baseline, dev, input_name, X, y, input_shape)
    logger.info("  TVM baseline accuracy: %.4f", tvm_accuracy_baseline)

    # ── Optional MetaSchedule tuning ──────────────────────────────────────────
    tvm_latency_tuned  = None
    tvm_accuracy_tuned = None
    tuned_path         = None

    if tune:
        work_dir   = output_dir / "tuning_logs"
        lib_tuned  = _tune_with_metaschedule(
            model_path, input_name, input_shape, target, work_dir, n_trials
        )
        tuned_path = output_dir / "model_tuned.so"
        lib_tuned.export_library(str(tuned_path))
        logger.info("Tuned library → %s  (%.1f KB)",
                    tuned_path.name, tuned_path.stat().st_size / 1024)
        tvm_latency_tuned  = _tvm_latency(lib_tuned, dev, input_name, sample)
        tvm_accuracy_tuned = _tvm_accuracy(lib_tuned, dev, input_name, X, y, input_shape)
        logger.info("  TVM tuned latency:  %.3f ms  (speedup %.2fx vs ONNX)",
                    tvm_latency_tuned, onnx_latency / tvm_latency_tuned)
        logger.info("  TVM tuned accuracy: %.4f", tvm_accuracy_tuned)

    # ── MLflow logging ────────────────────────────────────────────────────────
    try:
        import mlflow
        uri = mlflow_uri or "mlruns/"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        run_name = f"tvm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_path",  str(model_path))
            mlflow.log_param("target",      target_str)
            mlflow.log_param("opt_level",   opt_level)
            mlflow.log_param("tuned",       tune)
            mlflow.log_metric("onnx_latency_ms",         onnx_latency)
            mlflow.log_metric("tvm_latency_baseline_ms", tvm_latency_baseline)
            mlflow.log_metric("tvm_accuracy_baseline",   tvm_accuracy_baseline)
            mlflow.log_metric("speedup_baseline",        onnx_latency / tvm_latency_baseline)
            if tune and tvm_latency_tuned:
                mlflow.log_metric("tvm_latency_tuned_ms", tvm_latency_tuned)
                mlflow.log_metric("tvm_accuracy_tuned",   tvm_accuracy_tuned)
                mlflow.log_metric("speedup_tuned",        onnx_latency / tvm_latency_tuned)
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "timestamp":               datetime.now().isoformat(timespec="seconds"),
        "model_path":              str(model_path),
        "features_dir":            str(features_dir),
        "target":                  target_str,
        "opt_level":               opt_level,
        "input_name":              input_name,
        "input_shape":             input_shape,
        "n_val_samples":           len(y),
        "label_names":             label_names,
        "onnx_latency_ms":         onnx_latency,
        "tvm_latency_baseline_ms": tvm_latency_baseline,
        "tvm_accuracy_baseline":   tvm_accuracy_baseline,
        "speedup_baseline":        onnx_latency / tvm_latency_baseline,
        "baseline_library":        str(baseline_path),
        "tuned":                   tune,
        "tvm_latency_tuned_ms":    tvm_latency_tuned,
        "tvm_accuracy_tuned":      tvm_accuracy_tuned,
        "speedup_tuned":           (onnx_latency / tvm_latency_tuned) if tvm_latency_tuned else None,
        "tuned_library":           str(tuned_path) if tuned_path else None,
    }
    report_path = output_dir / "tvm_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("TVM report → %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.compilation.compile_tvm",
        description="Stage 6 — TVM model compilation and benchmarking.",
    )
    parser.add_argument("--config",       metavar="YAML")
    parser.add_argument("--model",        metavar="PATH")
    parser.add_argument("--features-val",  metavar="DIR")
    parser.add_argument("--output",       metavar="DIR")
    parser.add_argument("--class-filter", metavar="CLASS", nargs="+")
    parser.add_argument("--target",       default="llvm")
    parser.add_argument("--opt-level",    type=int, default=3)
    parser.add_argument("--tune",         action="store_true")
    parser.add_argument("--n-trials",     type=int, default=400)
    parser.add_argument("--experiment",   default="tvm-compilation")
    parser.add_argument("--mlflow-uri",   default=None)
    args = parser.parse_args(argv)

    cfg: dict = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    def _get(key, cli_val, default=None):
        return cli_val if cli_val is not None else cfg.get(key, default)

    model_path   = Path(_get("model",    args.model))
    features_dir = Path(_get("features_val", args.features_val))
    output_dir   = Path(_get("output",   args.output))
    class_filter = args.class_filter or cfg.get("class_filter") or None
    target_str   = str(_get("target",    args.target,    "llvm"))
    opt_level    = int(_get("opt_level", args.opt_level, 3))
    tune         = bool(_get("tune",     args.tune,      False))
    n_trials     = int(_get("n_trials",  args.n_trials,  400))
    experiment   = str(_get("experiment",args.experiment,"tvm-compilation"))
    mlflow_uri   = _get("mlflow_uri",    args.mlflow_uri)

    for label, p in [("model", model_path), ("features-val", features_dir)]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            sys.exit(1)

    logger.info("Model:        %s", model_path)
    logger.info("Features-val: %s", features_dir)
    logger.info("Output:       %s", output_dir)
    logger.info("Target:       %s", target_str)
    logger.info("Class filter: %s", class_filter or "(none)")
    logger.info("Tune:         %s  (trials=%d)", tune, n_trials)

    report = compile_model(
        model_path   = model_path,
        features_dir = features_dir,
        output_dir   = output_dir,
        class_filter = class_filter,
        target_str   = target_str,
        opt_level    = opt_level,
        tune         = tune,
        n_trials     = n_trials,
        experiment   = experiment,
        mlflow_uri   = mlflow_uri,
    )

    logger.info("═" * 56)
    logger.info("ONNX RT baseline: %.3f ms", report["onnx_latency_ms"])
    logger.info("TVM baseline:     %.3f ms  (%.2fx speedup)",
                report["tvm_latency_baseline_ms"], report["speedup_baseline"])
    if report["tuned"]:
        logger.info("TVM tuned:        %.3f ms  (%.2fx speedup)",
                    report["tvm_latency_tuned_ms"], report["speedup_tuned"])
    logger.info("Accuracy:         %.4f", report["tvm_accuracy_baseline"])
    logger.info("═" * 56)


if __name__ == "__main__":
    main()
