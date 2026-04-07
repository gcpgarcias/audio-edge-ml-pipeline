"""
Arena estimator for Keras sequential CNN models.

Computes the peak scratch-RAM required during inference using a two-buffer
(ping-pong) strategy: at any point only the current layer's input tensor and
output tensor need to be alive simultaneously.

Arena bytes = max over all layers of (input_bytes + output_bytes)

Weights are stored as const arrays in Flash and are NOT counted here.

Usage
-----
    from src.deployment.codegen.arena_estimator import estimate_arena

    layers = estimate_arena(model)
    for l in layers:
        print(l)
    peak = max(l.input_bytes + l.output_bytes for l in layers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ArenaLayer:
    index:        int
    name:         str
    kind:         str               # conv2d | maxpool | gap | dense | norm | dropout | input
    input_shape:  tuple
    output_shape: tuple
    weight_bytes: int               # Flash — not counted in arena
    input_bytes:  int               # arena ping  (float32)
    output_bytes: int               # arena pong  (float32)
    peak_bytes:   int               # input_bytes + output_bytes

    def __str__(self) -> str:
        w_kb  = self.weight_bytes / 1024
        pk_kb = self.peak_bytes   / 1024
        return (
            f"[{self.index:2d}] {self.name:<32}  "
            f"in={str(self.input_shape):<20}  "
            f"out={str(self.output_shape):<20}  "
            f"weights={w_kb:7.1f} KB  peak_arena={pk_kb:7.1f} KB"
        )


def _floats(shape: tuple) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _bytes(shape: tuple) -> int:
    return _floats(shape) * 4   # float32


def _conv_output_shape(input_shape: tuple, filters: int,
                       kernel: tuple, padding: str, strides: tuple) -> tuple:
    h, w, _ = input_shape
    if padding == "same":
        oh = math.ceil(h / strides[0])
        ow = math.ceil(w / strides[1])
    else:  # valid
        oh = math.ceil((h - kernel[0] + 1) / strides[0])
        ow = math.ceil((w - kernel[1] + 1) / strides[1])
    return (oh, ow, filters)


def _pool_output_shape(input_shape: tuple, pool: tuple, strides: Optional[tuple]) -> tuple:
    h, w, c = input_shape
    s = strides if strides else pool
    return (h // s[0], w // s[1], c)


def _gap_output_shape(input_shape: tuple) -> tuple:
    return (input_shape[-1],)


def _dense_output_shape(units: int) -> tuple:
    return (units,)


def estimate_arena(model) -> list[ArenaLayer]:
    """Walk a Keras model layer by layer and compute arena requirements.

    Supports: InputLayer, Normalization, Conv2D, MaxPooling2D,
              GlobalAveragePooling2D, Dense, Dropout.

    Parameters
    ----------
    model:
        A compiled (or at least built) Keras Model.

    Returns
    -------
    List of ArenaLayer, one per layer (InputLayer excluded from peak calc
    since the input tensor is provided by the caller).
    """
    import tensorflow as tf

    layers_out: list[ArenaLayer] = []
    current_shape: Optional[tuple] = None

    for idx, layer in enumerate(model.layers):
        cls  = layer.__class__.__name__
        cfg  = layer.get_config()
        w    = layer.get_weights()

        # ── Determine input / output shapes ──────────────────────────────────
        def _to_shape(tensor) -> tuple:
            raw = tensor.shape
            lst = raw.as_list() if hasattr(raw, "as_list") else list(raw)
            return tuple(d for d in lst if d is not None)

        try:
            out_shape = _to_shape(layer.output)
        except Exception:
            out_shape = None

        try:
            inp = layer.input
            # multi-input layers return a list/tuple of tensors
            if isinstance(inp, (list, tuple)):
                inp = inp[0]
            in_shape_cur = _to_shape(inp)
        except Exception:
            in_shape_cur = current_shape or (0,)

        in_shape = in_shape_cur or current_shape or (0,)

        # ── Weight bytes (Flash) ──────────────────────────────────────────
        weight_bytes = sum(arr.nbytes for arr in w)

        # ── Classify layer kind ───────────────────────────────────────────
        kind_map = {
            "InputLayer":              "input",
            "Normalization":           "norm",
            "Conv2D":                  "conv2d",
            "DepthwiseConv2D":         "dwconv2d",
            "SeparableConv2D":         "sepconv2d",
            "MaxPooling2D":            "maxpool",
            "AveragePooling2D":        "avgpool",
            "GlobalAveragePooling2D":  "gap",
            "GlobalMaxPooling2D":      "gmp",
            "Dense":                   "dense",
            "Dropout":                 "dropout",
            "Flatten":                 "flatten",
            "BatchNormalization":      "batchnorm",
            "Activation":              "activation",
            "Reshape":                 "reshape",
        }
        kind = kind_map.get(cls, cls.lower())

        in_bytes  = _bytes(in_shape) if in_shape != (0,) else 0
        out_bytes = _bytes(out_shape) if out_shape else 0
        peak      = in_bytes + out_bytes

        layers_out.append(ArenaLayer(
            index        = idx,
            name         = layer.name,
            kind         = kind,
            input_shape  = in_shape,
            output_shape = out_shape or (),
            weight_bytes = weight_bytes,
            input_bytes  = in_bytes,
            output_bytes = out_bytes,
            peak_bytes   = peak,
        ))

        if out_shape:
            current_shape = out_shape

    return layers_out


def check_max_ram(model, max_ram_kb: float, verbose: bool = True) -> tuple[int, list[ArenaLayer]]:
    """Estimate arena and check against a RAM budget.

    Parameters
    ----------
    model:
        Keras model.
    max_ram_kb:
        Maximum allowed arena in KB.
    verbose:
        Print per-layer table.

    Returns
    -------
    (peak_bytes, layers)

    Raises
    ------
    ValueError if peak_bytes > max_ram_kb * 1024.
    """
    layers   = estimate_arena(model)
    peak     = max((l.peak_bytes for l in layers if l.kind != "input"), default=0)
    total_w  = sum(l.weight_bytes for l in layers)
    peak_kb  = peak / 1024

    if verbose:
        print(f"\n{'Layer':<34}  {'Input shape':<20}  {'Output shape':<20}  "
              f"{'Weights':>10}  {'Peak arena':>12}")
        print("─" * 105)
        for l in layers:
            marker = " ◄ BOTTLENECK" if l.peak_bytes == peak and l.kind != "input" else ""
            w_kb   = l.weight_bytes / 1024
            pk_kb  = l.peak_bytes   / 1024
            print(f"[{l.index:2d}] {l.name:<30}  {str(l.input_shape):<20}  "
                  f"{str(l.output_shape):<20}  {w_kb:8.1f} KB  {pk_kb:10.1f} KB{marker}")
        print("─" * 105)
        print(f"{'Total weights (Flash):':<55}  {total_w/1024:8.1f} KB")
        print(f"{'Peak arena (RAM):':<55}  {peak_kb:8.1f} KB  "
              f"({'OK' if peak_kb <= max_ram_kb else 'EXCEEDS BUDGET'})")
        print(f"{'Budget:':<55}  {max_ram_kb:8.1f} KB\n")

    if peak_kb > max_ram_kb:
        bottleneck = max((l for l in layers if l.kind != "input"),
                         key=lambda l: l.peak_bytes)
        raise ValueError(
            f"Peak arena {peak_kb:.1f} KB exceeds --max-ram {max_ram_kb:.0f} KB.\n"
            f"Bottleneck: layer '{bottleneck.name}' ({bottleneck.kind}) "
            f"needs {bottleneck.peak_bytes/1024:.1f} KB "
            f"({bottleneck.input_shape} → {bottleneck.output_shape}).\n"
            f"Suggestions:\n"
            f"  • Reduce filters (current bottleneck has large output tensor)\n"
            f"  • Add stride=2 to the first Conv2D instead of MaxPool after it\n"
            f"  • Reduce input resolution (fewer mels, shorter duration, larger hop)\n"
            f"  • Use depthwise separable convolutions (DepthwiseConv2D)\n"
            f"  • Use int8 quantization (--dtype int8) to cut arena by 4x"
        )

    return peak, layers
