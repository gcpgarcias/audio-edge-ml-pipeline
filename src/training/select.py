"""
Stage 5 — Model Selection (two-checkpoint design).

Overview
--------
Selection runs at **two points** in the pipeline:

1. **Pre-optimisation** (automatic, end of Stage 3)
   Ranks all finished MLflow runs by a quality metric (e.g. ``val_f1_macro``),
   applies an optional accuracy floor, and writes a ``shortlist.json`` with the
   top-N candidates to pass to Stage 6 for optimisation.
   No size filter is applied here because Stage 6/7 will change model sizes.

2. **Post-optimisation** (manual, after Stage 6)
   Reads ``optimization_report.json`` files produced by Stage 6, applies the
   hard ``--max-size-kb`` constraint against *real* optimised sizes, ranks by
   a post-optimisation metric, and writes the final ``best_model.json``.

Public Python API
-----------------
Both functions below are imported by ``train.py`` for auto-selection::

    from src.training.select import select_preopt, write_shortlist

    candidates = select_preopt(experiment="birdeep", top_n=5)
    write_shortlist(candidates, Path("data/models/shortlist.json"), "birdeep")

CLI — pre-opt (default mode)
-----------------------------
::

    python -m src.training.select \\
        --experiment birdeep-classification \\
        [--min-accuracy 0.70] \\
        [--metric val_f1_macro] \\
        [--top-n 5] \\
        [--output data/models/shortlist.json]

CLI — post-opt mode
-------------------
::

    python -m src.training.select \\
        --post-opt \\
        --shortlist data/models/shortlist.json \\
        --opt-dir  data/models/optimized \\
        [--max-size-kb 256] \\
        [--metric val_accuracy_optimized] \\
        [--output data/models/best_model.json]

``optimization_report.json`` schema (Stage 6 contract)
-------------------------------------------------------
Stage 6 must write one file per model at
``<opt-dir>/<model_name>/optimization_report.json`` with the following fields::

    {
      "run_id":                 "abc123…",
      "run_name":               "birdeep_svm",
      "model_name":             "svm",
      "original_model_path":    "data/models/birdeep_svm/svm.joblib",
      "optimized_model_path":   "data/models/optimized/birdeep_svm/model.tflite",
      "original_size_kb":       400.0,
      "optimized_size_kb":      98.5,
      "compression_ratio":      4.06,
      "quantization_method":    "int8",
      "target_device":          "nicla_vision",
      "val_accuracy_original":  0.934,
      "val_accuracy_optimized": 0.921,
      "accuracy_drop":          0.013,
      "latency_ms":             null,
      "timestamp":              "2024-01-01T12:00:00"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _setup_mlflow(uri: Optional[str]):
    import mlflow
    tracking_uri = uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns/")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def _fetch_runs(mlflow, experiment_name: str) -> list[dict]:
    """Return dicts for all FINISHED runs in *experiment_name*."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.error("Experiment '%s' not found in MLflow.", experiment_name)
        logger.info("Run 'mlflow experiments list' to see available experiments.")
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=500,
    )

    records = []
    for run in runs:
        m = run.data.metrics
        p = run.data.params

        features_dir = p.get("features_dir")
        # Prefer an explicitly logged eval dir; fall back to convention
        # (_train → _val) if the derived path exists on disk.
        features_eval_dir = p.get("features_eval_dir")
        if features_eval_dir is None and features_dir:
            candidate_eval = features_dir.replace("_train", "_val")
            if candidate_eval != features_dir and Path(candidate_eval).exists():
                features_eval_dir = candidate_eval

        records.append({
            "run_id":             run.info.run_id,
            "run_name":           run.info.run_name or run.info.run_id[:8],
            "model":              p.get("model", "unknown"),
            "val_accuracy":       m.get("val_accuracy"),
            "val_f1_macro":       m.get("val_f1_macro"),
            "model_size_kb":      m.get("model_size_kb"),
            "params":             p,
            "metrics":            m,
            "artifact_uri":       run.info.artifact_uri,
            "features_dir":       features_dir,
            "features_eval_dir":  features_eval_dir,
            "class_filter":       p.get("class_filter"),
        })
    return records


def _rank_runs(
    records:      list[dict],
    metric:       str,
    min_accuracy: Optional[float],
) -> list[dict]:
    """Filter by accuracy floor (no size filter) and rank by *metric*."""
    survivors = []
    for r in records:
        if r.get("val_accuracy") is None:
            continue
        if min_accuracy is not None and (r["val_accuracy"] or 0.0) < min_accuracy:
            continue
        rank_val = r.get("metrics", {}).get(metric) or r.get(metric)
        if rank_val is None:
            continue
        r["_rank_metric"] = float(rank_val)
        survivors.append(r)
    survivors.sort(key=lambda r: r["_rank_metric"], reverse=True)
    return survivors


# ---------------------------------------------------------------------------
# Public API — callable from train.py
# ---------------------------------------------------------------------------

def select_preopt(
    experiment:   str,
    mlflow_uri:   Optional[str]  = None,
    metric:       str            = "val_f1_macro",
    min_accuracy: Optional[float] = None,
    top_n:        int            = 5,
) -> list[dict]:
    """Query MLflow, rank by *metric*, return the top *top_n* candidates.

    No size filter is applied — that happens post-optimisation when real
    compressed sizes are known.

    Parameters
    ----------
    experiment:
        MLflow experiment name.
    mlflow_uri:
        Tracking URI. *None* → ``MLFLOW_TRACKING_URI`` env var or ``"mlflow/"``.
    metric:
        MLflow scalar metric to rank by (descending).
    min_accuracy:
        Optional hard floor on ``val_accuracy``.
    top_n:
        Maximum shortlist length.

    Returns
    -------
    list[dict]
        Ranked records (may be shorter than *top_n* if fewer runs qualify).
    """
    mlflow = _setup_mlflow(mlflow_uri)
    records = _fetch_runs(mlflow, experiment)
    ranked = _rank_runs(records, metric, min_accuracy)
    return ranked[:top_n]


def write_shortlist(
    records:              list[dict],
    path:                 Path,
    experiment:           str,
    metric:               str = "val_f1_macro",
    features_eval_dir_override: Optional[str] = None,
) -> None:
    """Serialise *records* as ``shortlist.json`` at *path*.

    Parameters
    ----------
    records:
        Output of :func:`select_preopt`.
    path:
        Destination file path.
    experiment:
        Experiment name — stored in the JSON for traceability.
    metric:
        The metric used for ranking — stored in the JSON.
    """
    candidates = [
        {
            "rank":              i + 1,
            "run_id":            r["run_id"],
            "run_name":          r.get("run_name"),
            "model":             r.get("model"),
            "val_accuracy":      r.get("val_accuracy"),
            "val_f1_macro":      r.get("val_f1_macro"),
            "model_size_kb":     r.get("model_size_kb"),
            "params":            r.get("params", {}),
            "artifact_uri":      r.get("artifact_uri"),
            "features_dir":      r.get("features_dir"),
            "features_eval_dir": features_eval_dir_override or r.get("features_eval_dir"),
            "class_filter":      r.get("class_filter"),
        }
        for i, r in enumerate(records)
    ]
    out = {
        "experiment":   experiment,
        "metric":       metric,
        "n_candidates": len(candidates),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates":   candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("Shortlist (%d candidates) written: %s", len(candidates), path)


# ---------------------------------------------------------------------------
# Post-optimisation selection
# ---------------------------------------------------------------------------

def select_postopt(
    shortlist_path: Path,
    opt_dir:        Path,
    max_size_kb:    Optional[float] = None,
    metric:         str             = "val_accuracy_optimized",
    ascending:      bool            = False,
) -> Optional[dict]:
    """Read Stage 6 optimisation reports and return the best surviving model.

    Parameters
    ----------
    shortlist_path:
        Path to a ``shortlist.json`` produced by :func:`write_shortlist`.
    opt_dir:
        Directory containing one subdirectory per model, each with an
        ``optimization_report.json``.  Expected layout::

            opt_dir/
            └── <model_name>/
                └── optimization_report.json

    max_size_kb:
        Hard upper bound on ``optimized_size_kb``.  *None* → no size filter.
    metric:
        Field in ``optimization_report.json`` used for ranking (descending).
        Common values: ``"val_accuracy_optimized"``, ``"val_f1_macro_optimized"``.

    Returns
    -------
    dict or None
        The winning optimisation report dict, or *None* if nothing qualifies.
    """
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Shortlist not found: {shortlist_path}")

    shortlist = json.loads(shortlist_path.read_text()).get("candidates", [])
    if not shortlist:
        logger.warning("Shortlist is empty — nothing to evaluate.")
        return None

    results = []
    for candidate in shortlist:
        model_name = candidate.get("model", "unknown")
        run_name   = candidate.get("run_name") or model_name
        # optimize.py writes under run_name when it differs from model_name
        # (e.g. two CNN variants); fall back to model_name for older outputs.
        report_path = opt_dir / run_name / "optimization_report.json"
        if not report_path.exists():
            report_path = opt_dir / model_name / "optimization_report.json"
        if not report_path.exists():
            logger.warning("No optimization_report.json for '%s' — skipping.", run_name)
            continue

        report = json.loads(report_path.read_text())

        if max_size_kb is not None:
            opt_size = report.get("optimized_size_kb")
            if opt_size is not None and opt_size > max_size_kb:
                logger.debug(
                    "Filtered out '%s': optimized_size_kb=%.1f > %.1f",
                    model_name, opt_size, max_size_kb,
                )
                continue

        rank_val = report.get(metric)
        if rank_val is None:
            logger.debug("Metric '%s' missing in report for '%s' — skipping.", metric, model_name)
            continue

        report["_rank_metric"] = float(rank_val)
        report["_shortlist_candidate"] = candidate
        results.append(report)

    if not results:
        return None

    results.sort(key=lambda r: r["_rank_metric"], reverse=not ascending)
    return results[0]


# ---------------------------------------------------------------------------
# Tabular output helpers
# ---------------------------------------------------------------------------

def _fmt_float(v, width: int) -> str:
    if v is None:
        return "N/A".rjust(width)
    return f"{v:.4f}".rjust(width)


def _print_preopt_table(records: list[dict], metric: str, top_n: int) -> None:
    header = (
        f"{'#':>4}  "
        f"{'Model':<16}  "
        f"{'Run name':<32}  "
        f"{'Accuracy':>10}  "
        f"{'F1-macro':>10}  "
        f"{'Size (KB)':>10}  "
        f"{'Rank (' + metric[:12] + ')':>18}  "
        f"{'Run ID':<12}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for i, r in enumerate(records[:top_n], start=1):
        mark = " *" if i == 1 else "  "
        print(
            f"{i:>4}{mark}"
            f"{r.get('model', '?'):<16}  "
            f"{(r.get('run_name') or '')[:32]:<32}  "
            f"{_fmt_float(r.get('val_accuracy'), 10)}  "
            f"{_fmt_float(r.get('val_f1_macro'), 10)}  "
            f"{_fmt_float(r.get('model_size_kb'), 10)}  "
            f"{_fmt_float(r.get('_rank_metric'), 18)}  "
            f"{r['run_id'][:12]}"
        )
    print(sep)
    print(f"  * Shortlist #1  |  Top {min(top_n, len(records))} of {len(records)} qualifying run(s).")
    print()


def _print_postopt_table(results: list[dict], metric: str, ascending: bool = False) -> None:
    direction = "↑ asc" if ascending else "↓ desc"
    rank_hdr  = f"{'Rank (' + metric[:10] + ') ' + direction:>22}"
    header = (
        f"{'#':>4}  "
        f"{'Run name':<32}  "
        f"{'Opt size (KB)':>14}  "
        f"{'Orig size (KB)':>15}  "
        f"{'Acc (opt)':>10}  "
        f"{'Acc drop':>10}  "
        f"{rank_hdr}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for i, r in enumerate(results, start=1):
        mark    = " *" if i == 1 else "  "
        label   = r.get("run_name") or r.get("model_name", "?")
        print(
            f"{i:>4}{mark}"
            f"{label[:32]:<32}  "
            f"{_fmt_float(r.get('optimized_size_kb'), 14)}  "
            f"{_fmt_float(r.get('original_size_kb'), 15)}  "
            f"{_fmt_float(r.get('val_accuracy_optimized'), 10)}  "
            f"{_fmt_float(r.get('accuracy_drop'), 10)}  "
            f"{_fmt_float(r.get('_rank_metric'), 22)}"
        )
    print(sep)
    print(f"  * Best post-optimisation model  |  {len(results)} model(s) evaluated.")
    print()


# ---------------------------------------------------------------------------
# best_model.json writer
# ---------------------------------------------------------------------------

def _write_best(report: dict, path: Path, experiment: str) -> None:
    candidate = report.get("_shortlist_candidate", {})
    out = {
        "run_id":                 report.get("run_id") or candidate.get("run_id"),
        "run_name":               report.get("run_name") or candidate.get("run_name"),
        "model":                  report.get("model_name") or candidate.get("model"),
        "optimized_model_path":   report.get("optimized_model_path"),
        "original_size_kb":       report.get("original_size_kb"),
        "optimized_size_kb":      report.get("optimized_size_kb"),
        "compression_ratio":      report.get("compression_ratio"),
        "quantization_method":    report.get("quantization_method"),
        "val_accuracy_original":  report.get("val_accuracy_original"),
        "val_accuracy_optimized": report.get("val_accuracy_optimized"),
        "accuracy_drop":          report.get("accuracy_drop"),
        "latency_ms":             report.get("latency_ms"),
        "params":                 candidate.get("params", {}),
        "experiment":             experiment,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("Best model written: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.training.select",
        description="Stage 5 — Model Selection (two-checkpoint design)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--post-opt", action="store_true",
        help="Post-optimisation mode: read optimization_report.json files from Stage 6.",
    )

    # ── Pre-opt flags ──────────────────────────────────────────────────────
    pre = p.add_argument_group("pre-optimisation flags (default mode)")
    pre.add_argument(
        "--experiment", default="ml-pipeline",
        help="MLflow experiment name (default: ml-pipeline).",
    )
    pre.add_argument(
        "--min-accuracy", type=float, metavar="FLOAT",
        help="Hard lower bound on val_accuracy (e.g. 0.70).",
    )
    pre.add_argument(
        "--metric", default="val_f1_macro",
        help="Metric to rank by (default: val_f1_macro).",
    )
    pre.add_argument(
        "--top-n", type=int, default=5, metavar="N",
        help="Shortlist size (default: 5).",
    )
    pre.add_argument(
        "--mlflow-uri", metavar="URI",
        help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI or 'mlflow/').",
    )
    pre.add_argument(
        "--features-eval-dir", metavar="DIR", dest="features_eval_dir",
        help=(
            "Override the features_eval_dir field for every candidate in the "
            "shortlist.  Useful when the held-out set is in a non-standard "
            "location.  When omitted, select.py auto-derives the eval dir from "
            "features_dir by replacing '_train' with '_val' (if that path "
            "exists), or reads 'features_eval_dir' from the MLflow run params."
        ),
    )

    # ── Post-opt flags ─────────────────────────────────────────────────────
    post = p.add_argument_group("post-optimisation flags (--post-opt mode)")
    post.add_argument(
        "--shortlist", metavar="PATH",
        help="Path to shortlist.json written by Stage 3.",
    )
    post.add_argument(
        "--opt-dir", metavar="DIR",
        help="Directory containing <model>/optimization_report.json files.",
    )
    post.add_argument(
        "--max-size-kb", type=float, metavar="FLOAT",
        help="Hard upper bound on optimized_size_kb (post-opt only).",
    )
    post.add_argument(
        "--sort-asc", action="store_true", default=False,
        help=(
            "Sort ascending instead of descending.  Use this when --metric "
            "is a cost (e.g. latency_ms, onnx_latency_ms) where lower is better."
        ),
    )

    # ── Shared output ──────────────────────────────────────────────────────
    p.add_argument(
        "--output", metavar="PATH",
        help=(
            "Output path.  Defaults to 'data/models/shortlist.json' (pre-opt) "
            "or 'data/models/best_model.json' (post-opt)."
        ),
    )

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Guard: --max-size-kb only valid in post-opt mode
    if args.max_size_kb is not None and not args.post_opt:
        parser.error(
            "--max-size-kb is only valid in --post-opt mode.\n"
            "In pre-optimisation selection, sizes are unknown until Stage 6 runs.\n"
            "Use --min-accuracy to set a quality floor instead."
        )

    # ── Post-optimisation mode ─────────────────────────────────────────────
    if args.post_opt:
        if not args.shortlist:
            parser.error("--post-opt requires --shortlist <path>")
        if not args.opt_dir:
            parser.error("--post-opt requires --opt-dir <directory>")

        shortlist_path = Path(args.shortlist)
        opt_dir        = Path(args.opt_dir)
        output_path    = Path(args.output or "data/models/best_model.json")

        # Load shortlist to get experiment name for output metadata
        shortlist_data  = json.loads(shortlist_path.read_text())
        experiment_name = shortlist_data.get("experiment", "unknown")

        logger.info("Post-opt selection from %s …", opt_dir)
        best = select_postopt(
            shortlist_path = shortlist_path,
            opt_dir        = opt_dir,
            max_size_kb    = args.max_size_kb,
            metric         = args.metric,
            ascending      = args.sort_asc,
        )

        if best is None:
            logger.error(
                "No models survive the post-opt constraints "
                "(max_size_kb=%s, metric=%s).",
                args.max_size_kb, args.metric,
            )
            sys.exit(1)

        # Re-collect the full ranked list without size filter for the display table
        candidates   = shortlist_data.get("candidates", [])
        display_list = []
        for c in candidates:
            _run  = c.get("run_name") or c.get("model") or ""
            _mod  = c.get("model") or ""
            rp = opt_dir / _run / "optimization_report.json"
            if not rp.exists():
                rp = opt_dir / _mod / "optimization_report.json"
            if rp.exists():
                r = json.loads(rp.read_text())
                rv = r.get(args.metric)
                if rv is not None:
                    r["_rank_metric"] = float(rv)
                    display_list.append(r)
        display_list.sort(key=lambda r: r["_rank_metric"], reverse=not args.sort_asc)

        _print_postopt_table(display_list, args.metric, ascending=args.sort_asc)
        _write_best(best, output_path, experiment_name)

        opt_size = best.get("optimized_size_kb", 0) or 0
        print(
            f"Best model: {best.get('model_name', '?')}  "
            f"optimized_size={opt_size:.1f} KB  "
            f"{args.metric}={best.get('_rank_metric', float('nan')):.4f}"
        )
        return

    # ── Pre-optimisation mode (default) ───────────────────────────────────
    output_path = Path(args.output or "data/models/shortlist.json")

    logger.info("Fetching runs for experiment '%s' …", args.experiment)
    candidates = select_preopt(
        experiment   = args.experiment,
        mlflow_uri   = args.mlflow_uri,
        metric       = args.metric,
        min_accuracy = args.min_accuracy,
        top_n        = args.top_n,
    )

    if not candidates:
        logger.error(
            "No qualifying runs found "
            "(experiment=%s, min_accuracy=%s, metric=%s).",
            args.experiment, args.min_accuracy, args.metric,
        )
        logger.info("Run 'python -m src.training.train' first.")
        sys.exit(1)

    _print_preopt_table(candidates, args.metric, args.top_n)
    write_shortlist(
        candidates, output_path, args.experiment, args.metric,
        features_eval_dir_override=getattr(args, "features_eval_dir", None),
    )

    best = candidates[0]
    print(
        f"Shortlist #1: {best.get('model', '?')}  "
        f"(run_id={best['run_id'][:12]}…  "
        f"val_accuracy={best.get('val_accuracy', float('nan')):.4f}  "
        f"{args.metric}={best.get('_rank_metric', float('nan')):.4f})"
    )


if __name__ == "__main__":
    main()