"""
ONNX CNN → C codegen for embedded deployment.

Generates the same PlatformIO project layout as ModelToC, but reads
weights and graph topology from an ONNX file instead of a Keras model.
All tensors are assumed NCHW (the layout produced by tf2onnx).

Supported ONNX op patterns
---------------------------
    Sub + Mul                       → normalize (mean/scale constants)
    Reshape                         → shape tracking only
    Conv  [+ Relu]                  → ml_conv2d_nchw_relu / _linear
    MaxPool                         → ml_maxpool2d_nchw
    GlobalAveragePool + Squeeze     → ml_gap2d_nchw
    MatMul + Add  [+ Relu|Softmax]  → ml_dense_relu / _softmax / _linear
    Dropout                         → identity (skipped)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# C primitives — NCHW layout
# ---------------------------------------------------------------------------

_C_PRIMITIVES_NCHW = r"""
/* -------------------------------------------------------------------------
 * Inference layer primitives — pure C99, NCHW layout.
 * Tensor access: in[c * H * W + h * W + w]
 * ------------------------------------------------------------------------- */
#include <math.h>
#include <string.h>

/* ── Normalize (element-wise): out = (in - mean) * scale ─────────────────── */
static void ml_normalize(
    const float *in, int n, const float *mean, const float *scale, float *out)
{
    for (int i = 0; i < n; i++) out[i] = (in[i] - mean[0]) * scale[0];
}

/* ── Conv2D NCHW, asymmetric padding, relu activation ───────────────────── */
/* w layout: (out_c, in_c, kh, kw)                                           */
/* pads: [pad_h_top, pad_w_left, pad_h_bottom, pad_w_right]  (ONNX order)   */
static void ml_conv2d_nchw_relu(
    const float *in,  int in_c, int in_h, int in_w,
    const float *w,   int kh,   int kw,   int out_c,
    int stride_h, int stride_w,
    int pad_ht, int pad_wl, int pad_hb, int pad_wr,
    const float *b,
    float *out)
{
    int out_h = (in_h + pad_ht + pad_hb - kh) / stride_h + 1;
    int out_w = (in_w + pad_wl + pad_wr - kw) / stride_w + 1;
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = b[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int krow = 0; krow < kh; krow++) {
                        int ih = oh * stride_h + krow - pad_ht;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kcol = 0; kcol < kw; kcol++) {
                            int iw = ow * stride_w + kcol - pad_wl;
                            if (iw < 0 || iw >= in_w) continue;
                            acc += in[ic * in_h * in_w + ih * in_w + iw]
                                 * w[oc * in_c * kh * kw + ic * kh * kw + krow * kw + kcol];
                        }
                    }
                }
                out[oc * out_h * out_w + oh * out_w + ow] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* ── Conv2D NCHW, asymmetric padding, no activation ─────────────────────── */
static void ml_conv2d_nchw_linear(
    const float *in,  int in_c, int in_h, int in_w,
    const float *w,   int kh,   int kw,   int out_c,
    int stride_h, int stride_w,
    int pad_ht, int pad_wl, int pad_hb, int pad_wr,
    const float *b,
    float *out)
{
    int out_h = (in_h + pad_ht + pad_hb - kh) / stride_h + 1;
    int out_w = (in_w + pad_wl + pad_wr - kw) / stride_w + 1;
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = b[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int krow = 0; krow < kh; krow++) {
                        int ih = oh * stride_h + krow - pad_ht;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kcol = 0; kcol < kw; kcol++) {
                            int iw = ow * stride_w + kcol - pad_wl;
                            if (iw < 0 || iw >= in_w) continue;
                            acc += in[ic * in_h * in_w + ih * in_w + iw]
                                 * w[oc * in_c * kh * kw + ic * kh * kw + krow * kw + kcol];
                        }
                    }
                }
                out[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
}

/* ── MaxPool NCHW ─────────────────────────────────────────────────────────── */
static void ml_maxpool2d_nchw(
    const float *in, int in_c, int in_h, int in_w,
    int ph, int pw, int sh, int sw,
    float *out)
{
    int out_h = (in_h - ph) / sh + 1;
    int out_w = (in_w - pw) / sw + 1;
    for (int c = 0; c < in_c; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float mx = -3.402823466e+38f;
                for (int i = 0; i < ph; i++) {
                    for (int j = 0; j < pw; j++) {
                        float v = in[c * in_h * in_w + (oh*sh+i) * in_w + (ow*sw+j)];
                        if (v > mx) mx = v;
                    }
                }
                out[c * out_h * out_w + oh * out_w + ow] = mx;
            }
        }
    }
}

/* ── GlobalAveragePool NCHW → (out_c,) ───────────────────────────────────── */
static void ml_gap2d_nchw(
    const float *in, int in_c, int in_h, int in_w, float *out)
{
    float norm = 1.0f / (float)(in_h * in_w);
    for (int c = 0; c < in_c; c++) {
        float s = 0.0f;
        for (int i = 0; i < in_h * in_w; i++) s += in[c * in_h * in_w + i];
        out[c] = s * norm;
    }
}

/* ── Dense relu ─────────────────────────────────────────────────────────── */
/* w layout (ONNX MatMul): (in_features, out_features)                       */
static void ml_dense_relu(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    for (int o = 0; o < n_out; o++) {
        float acc = b[o];
        for (int i = 0; i < n_in; i++) acc += in[i] * w[i * n_out + o];
        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

/* ── Dense softmax ──────────────────────────────────────────────────────── */
static void ml_dense_softmax(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    float mx = -3.402823466e+38f;
    for (int o = 0; o < n_out; o++) {
        float acc = b[o];
        for (int i = 0; i < n_in; i++) acc += in[i] * w[i * n_out + o];
        out[o] = acc;
        if (acc > mx) mx = acc;
    }
    float s = 0.0f;
    for (int o = 0; o < n_out; o++) { out[o] = expf(out[o] - mx); s += out[o]; }
    for (int o = 0; o < n_out; o++) out[o] /= s;
}

/* ── Dense linear ───────────────────────────────────────────────────────── */
static void ml_dense_linear(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    for (int o = 0; o < n_out; o++) {
        float acc = b[o];
        for (int i = 0; i < n_in; i++) acc += in[i] * w[i * n_out + o];
        out[o] = acc;
    }
}
"""


# ---------------------------------------------------------------------------
# OnnxToC
# ---------------------------------------------------------------------------

class OnnxToC:
    """Convert an ONNX model (fp32) to a PlatformIO C project.

    Parameters
    ----------
    onnx_path:
        Path to the .onnx file.
    output_dir:
        Directory to write the project into.
    board:
        Board name from model_to_c.BOARDS catalogue.
    label_names:
        Ordered list of class name strings.
    feature_params:
        Dict with keys: sample_rate, n_fft, hop_length, n_mels, n_mfcc,
        duration, n_features.
    max_ram_kb:
        Raise if estimated peak arena exceeds this.
    """

    def __init__(
        self,
        onnx_path:      Path,
        output_dir:     Path,
        board:          str,
        label_names:    list[str],
        feature_params: dict,
        max_ram_kb:     float = 900.0,
    ) -> None:
        import onnx
        from onnx import numpy_helper, shape_inference
        from .model_to_c import BOARDS

        self.onnx_path  = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.board_cfg  = BOARDS[board]
        self.board_name = board
        self.labels     = label_names
        self.fp         = feature_params
        self.max_ram_kb = max_ram_kb

        raw   = onnx.load(str(self.onnx_path))
        self.model = shape_inference.infer_shapes(raw)

        # Build lookup maps
        self.init_map: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i)
            for i in self.model.graph.initializer
        }
        self.shape_map: dict[str, list[int]] = {}
        for vi in (list(self.model.graph.input)
                   + list(self.model.graph.value_info)
                   + list(self.model.graph.output)):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            self.shape_map[vi.name] = dims

        # Build dequantized weight map: resolve DequantizeLinear outputs → float32.
        # In static-int8 ONNX every weight tensor is wrapped in a DequantizeLinear
        # node whose output name is what Conv/MatMul refer to as their weight input.
        self.dequant_map: dict[str, np.ndarray] = {}
        for node in self.model.graph.node:
            if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
                q_name    = node.input[0]
                scale_name = node.input[1]
                zp_name    = node.input[2] if len(node.input) > 2 else None
                out_name   = node.output[0]
                if q_name in self.init_map and scale_name in self.init_map:
                    q   = self.init_map[q_name].astype(np.float32)
                    sc  = self.init_map[scale_name].astype(np.float32)
                    zp  = (self.init_map[zp_name].astype(np.float32)
                           if zp_name and zp_name in self.init_map else 0.0)
                    self.dequant_map[out_name] = (q - zp) * sc

    # ------------------------------------------------------------------

    def generate(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        src = self.output_dir / "src"
        src.mkdir(exist_ok=True)
        inc = self.output_dir / "include"
        inc.mkdir(exist_ok=True)

        print("Estimating arena ...")
        peak_bytes = self._estimate_arena()

        print("Generating weights ...")
        self._gen_weights(src, inc)

        print("Generating feature data ...")
        self._gen_feat_data(src, inc)

        print("Generating model forward pass ...")
        self._gen_model(src, inc, peak_bytes)

        print("Generating feature extraction ...")
        self._gen_features(src, inc)

        print("Generating audio driver ...")
        self._gen_audio(src, inc)

        print("Generating main.cpp ...")
        self._gen_main(src, peak_bytes)

        print("Generating platformio.ini ...")
        self._gen_platformio_ini()

        total = sum(p.stat().st_size for p in self.output_dir.rglob("*") if p.is_file())
        print(f"\nProject written to {self.output_dir}/")
        print(f"Total project size: {total/1024:.1f} KB")
        print(f"\nTo build and flash:")
        print(f"  cd {self.output_dir} && pio run --target upload")

    # ------------------------------------------------------------------
    # Arena estimation
    # ------------------------------------------------------------------

    def _tensor_bytes(self, name: str) -> int:
        shape = self.shape_map.get(name, [])
        if not shape:
            return 0
        n = 1
        for d in shape:
            if d and d > 0:
                n *= d
        return n * 4  # float32

    def _estimate_arena(self) -> int:
        """Estimate peak RAM needed for ping-pong activation buffers.

        Only compute-ops contribute to the arena (Conv, Pool, MatMul, etc.).
        DequantizeLinear and other weight-prep nodes are skipped because their
        outputs are weight tensors already stored as initializers, not activations.
        Peak = max(in_bytes + out_bytes) across all compute ops.
        """
        skip_ops = {"Dropout", "Squeeze", "Flatten", "Reshape",
                    "DequantizeLinear", "QuantizeLinear"}
        compute_ops = {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool",
                       "MatMul", "Gemm", "Sub", "Mul", "Add", "Softmax"}
        # Relu is applied in-place — no second buffer needed, exclude from peak
        peak = 0
        peak_node = ""
        for node in self.model.graph.node:
            if node.op_type not in compute_ops:
                continue
            # inputs that are activations (not weights/initializers)
            in_b = max(
                (self._tensor_bytes(t) for t in node.input
                 if t and t not in self.init_map and t not in self.dequant_map),
                default=0)
            out_b = max(
                (self._tensor_bytes(t) for t in node.output if t),
                default=0)
            layer_peak = in_b + out_b
            if layer_peak > peak:
                peak = layer_peak
                peak_node = f"{node.op_type} in={in_b//1024}KB out={out_b//1024}KB"

        peak_kb = peak / 1024
        print(f"  Peak arena:        {peak_kb:.1f} KB  (bottleneck: {peak_node})")
        print(f"  Budget:            {self.max_ram_kb:.0f} KB")

        if peak_kb > self.max_ram_kb:
            raise ValueError(
                f"Peak arena {peak_kb:.1f} KB exceeds --max-ram {self.max_ram_kb:.0f} KB. "
                f"Consider reducing filters, using fewer mels, or a shorter duration."
            )
        return peak

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _gen_weights(self, src: Path, inc: Path) -> None:
        decls: list[str] = []
        defs:  list[str] = []

        def _flt(v) -> str:
            s = f"{v:.8g}"
            if "." not in s and "e" not in s and "n" not in s:
                s += ".0"
            return s + "f"

        def _emit(name: str, arr: np.ndarray) -> None:
            arr  = arr.astype(np.float32)
            safe = name.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")
            flat = arr.flatten()
            vals = ", ".join(_flt(v) for v in flat)
            decls.append(f"extern const float wt_{safe}[{len(flat)}];  "
                         f"/* shape={arr.shape} */")
            defs.append(f"const float wt_{safe}[{len(flat)}] = {{\n"
                        + textwrap.fill(vals, width=100,
                                        initial_indent="    ",
                                        subsequent_indent="    ")
                        + "\n};\n")

        # Emit dequantized weights first (they shadow the raw int8 initializers)
        emitted = set()
        for name, arr in self.dequant_map.items():
            _emit(name, arr)
            emitted.add(name)

        # Emit remaining float initializers (norm constants, biases, etc.)
        for name, arr in self.init_map.items():
            if name in emitted:
                continue
            if arr.dtype.kind in ("i", "u") and arr.dtype != np.float32:
                continue  # skip int8 quantized tensors — already handled above
            _emit(name, arr)

        header = ("/* weights.h — ONNX model weights as const float arrays in Flash */\n"
                  "#pragma once\n\n"
                  + "\n".join(decls) + "\n")
        source = ('/* weights.c — generated by deploy.py (ONNX path) */\n'
                  '#include "weights.h"\n\n'
                  + "\n".join(defs))

        (inc / "weights.h").write_text(header)
        (src / "weights.c").write_text(source)

    # ------------------------------------------------------------------
    # Forward pass codegen
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> np.ndarray:
        """Return float32 array for a weight tensor name, resolving through DequantizeLinear."""
        if name in self.init_map:
            return self.init_map[name].astype(np.float32)
        if name in self.dequant_map:
            return self.dequant_map[name]
        raise KeyError(f"Weight '{name}' not found in initializers or dequant map")

    def _wref(self, name: str) -> str:
        safe = name.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")
        return f"wt_{safe}"

    def _gen_model(self, src: Path, inc: Path, peak_bytes: int) -> None:
        n_classes  = len(self.labels)
        arena_kb   = peak_bytes / 1024
        model_name = self.onnx_path.stem

        nodes    = list(self.model.graph.node)
        calls:   list[str] = []
        i        = 0
        cur_shape: list[int] = []  # NCHW shape of tensor currently in in_ptr

        # Initial input shape from graph input (skip batch dim)
        graph_in = self.model.graph.input[0]
        in_dims  = [d.dim_value for d in graph_in.type.tensor_type.shape.dim]
        # in_dims is NHWC: [0, H, W, C] or after reshape NCHW
        n_input  = 1
        for d in in_dims[1:]:
            if d and d > 0: n_input *= d

        # buf_b offset = max input bytes among ops that produce a new output buffer
        # (i.e. ops that call swap). In-place ops like Relu and no-ops don't swap,
        # so their input size doesn't determine buf_b placement.
        swapping_ops = {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool",
                        "MatMul", "Gemm", "Sub", "Mul", "Add", "Softmax",
                        "Normalization", "BatchNormalization"}
        skip_weight = self.init_map.keys() | self.dequant_map.keys()
        max_in = 0
        for node in nodes:
            if node.op_type not in swapping_ops:
                continue
            in_b = max((self._tensor_bytes(t) for t in node.input
                        if t and t not in skip_weight), default=0)
            if in_b > max_in: max_in = in_b

        max_in_floats = max_in // 4
        calls.append(f"    float *buf_a = arena;")
        calls.append(f"    float *buf_b = arena + {max_in_floats};")
        calls.append(f"    memcpy(buf_a, input, {n_input} * sizeof(float));")
        calls.append(f"    float *in_ptr = buf_a, *out_ptr = buf_b;")
        calls.append("")

        def swap():
            calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

        def peek_next_op(idx: int) -> Optional[str]:
            if idx + 1 < len(nodes):
                return nodes[idx + 1].op_type
            return None

        def get_attrs(node) -> dict:
            return {a.name: a for a in node.attribute}

        while i < len(nodes):
            node = nodes[i]
            op   = node.op_type

            if op in ("Dropout",):
                calls.append(f"    /* {node.output[0]}: Dropout — identity at inference */")
                i += 1

            elif op == "QuantizeLinear":
                # Activation quantize — input is a live tensor, not a weight.
                # We compute in float32, so treat as identity (no data movement).
                # Consume this node plus the paired DequantizeLinear that follows.
                calls.append(f"    /* QuantizeLinear+DequantizeLinear on activation — identity in fp32 codegen */")
                if peek_next_op(i) == "DequantizeLinear":
                    i += 2
                else:
                    i += 1

            elif op == "Sub":
                # Sub + Mul → normalize
                if i + 1 < len(nodes) and nodes[i + 1].op_type == "Mul":
                    mul_node = nodes[i + 1]
                    mean_name  = node.input[1]
                    # In static int8, the mul constant may be the non-data input
                    scale_name = (mul_node.input[1]
                                  if mul_node.input[0] == node.output[0]
                                  else mul_node.input[0])
                    out_shape  = self.shape_map.get(mul_node.output[0], [])
                    n_el = 1
                    for d in out_shape[1:]:
                        if d and d > 0: n_el *= d
                    calls.append(f"    /* normalize: (x - mean) * scale, n={n_el} */")
                    calls.append(f"    ml_normalize(in_ptr, {n_el}, "
                                 f"{self._wref(mean_name)}, {self._wref(scale_name)}, out_ptr);")
                    swap()
                    cur_shape = out_shape
                    i += 2
                else:
                    calls.append(f"    /* WARNING: standalone Sub — skipped */")
                    i += 1

            elif op == "Reshape":
                out_shape = self.shape_map.get(node.output[0], [])
                cur_shape = out_shape
                calls.append(f"    /* Reshape → {out_shape[1:]} (NCHW, no data movement) */")
                calls.append(f"    /* in_ptr already contiguous — reinterpret only */")
                i += 1

            elif op == "Conv":
                attrs      = get_attrs(node)
                strides    = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
                pads       = list(attrs["pads"].ints)    if "pads"    in attrs else [0, 0, 0, 0]
                w_name     = node.input[1]
                b_name     = node.input[2] if len(node.input) > 2 else None
                w_arr      = self._resolve(w_name)
                oc, ic, kh, kw = w_arr.shape
                # Input shape: try the node's input tensor, fall back to weight-derived ic
                in_shape   = self.shape_map.get(node.input[0], [])
                if not in_shape:
                    # shape_map may use the pre-dequant name; derive from weights
                    in_shape = [0, ic, 0, 0]
                in_h = in_shape[2] if len(in_shape) > 2 and in_shape[2] else 0
                in_w = in_shape[3] if len(in_shape) > 3 and in_shape[3] else 0
                # If shape is unknown, derive from output shape
                if in_h == 0 or in_w == 0:
                    out_shape = self.shape_map.get(node.output[0], [])
                    sh, sw = strides
                    # pads: [h_top, w_left, h_bottom, w_right]
                    ph_t = pads[0]; pw_l = pads[1]
                    ph_b = pads[2] if len(pads) > 2 else pads[0]
                    pw_r = pads[3] if len(pads) > 3 else pads[1]
                    if len(out_shape) > 3 and out_shape[2] and out_shape[3]:
                        in_h = out_shape[2] * sh - (ph_t + ph_b) + (kh - sh)
                        in_w = out_shape[3] * sw - (pw_l + pw_r) + (kw - sw)

                # pads: [h_top, w_left, h_bottom, w_right]
                ph_t = pads[0]; pw_l = pads[1]
                ph_b = pads[2] if len(pads) > 2 else pads[0]
                pw_r = pads[3] if len(pads) > 3 else pads[1]
                bias_ref = self._wref(b_name) if b_name else "NULL"
                calls.append(
                    f"    /* Conv {in_shape[1:]} → oc={oc} stride={strides} pad={pads} */")
                calls.append(
                    f"    ml_conv2d_nchw_linear(in_ptr, {ic}, {in_h}, {in_w}, "
                    f"{self._wref(w_name)}, {kh}, {kw}, {oc}, "
                    f"{strides[0]}, {strides[1]}, {ph_t}, {pw_l}, {ph_b}, {pw_r}, "
                    f"{bias_ref}, out_ptr);")
                swap()
                i += 1

            elif op == "Relu":
                # Apply in-place — no buffer swap needed
                out_shape = self.shape_map.get(node.input[0], [])
                n_el = 1
                for d in out_shape[1:]:
                    if d and d > 0: n_el *= d
                calls.append(f"    /* Relu in-place, n={n_el} */")
                calls.append(f"    for (int _r = 0; _r < {n_el}; _r++) "
                             f"if (in_ptr[_r] < 0.0f) in_ptr[_r] = 0.0f;")
                i += 1

            elif op == "MaxPool":
                attrs   = get_attrs(node)
                ks      = list(attrs["kernel_shape"].ints)
                strides = list(attrs["strides"].ints) if "strides" in attrs else ks
                in_shape = self.shape_map.get(node.input[0], [])
                in_c = in_shape[1] if len(in_shape) > 1 else 0
                in_h = in_shape[2] if len(in_shape) > 2 else 0
                in_w = in_shape[3] if len(in_shape) > 3 else 0
                calls.append(f"    /* MaxPool {in_shape[1:]} kernel={ks} stride={strides} */")
                calls.append(
                    f"    ml_maxpool2d_nchw(in_ptr, {in_c}, {in_h}, {in_w}, "
                    f"{ks[0]}, {ks[1]}, {strides[0]}, {strides[1]}, out_ptr);")
                swap()
                i += 1

            elif op == "GlobalAveragePool":
                in_shape = self.shape_map.get(node.input[0], [])
                in_c = in_shape[1] if len(in_shape) > 1 else 0
                in_h = in_shape[2] if len(in_shape) > 2 else 0
                in_w = in_shape[3] if len(in_shape) > 3 else 0
                calls.append(f"    /* GlobalAveragePool {in_shape[1:]} → ({in_c},) */")
                calls.append(
                    f"    ml_gap2d_nchw(in_ptr, {in_c}, {in_h}, {in_w}, out_ptr);")
                swap()
                # consume trailing Squeeze if present
                if peek_next_op(i) == "Squeeze":
                    calls.append(f"    /* Squeeze — no-op (GAP already 1D) */")
                    i += 2
                else:
                    i += 1

            elif op == "Squeeze":
                calls.append(f"    /* Squeeze — no-op */")
                i += 1

            elif op == "MatMul":
                # MatMul + Add [+ Relu | Softmax]
                if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                    add_node = nodes[i + 1]
                    w_name   = node.input[1]
                    b_name   = add_node.input[1]
                    w_arr    = self._resolve(w_name)
                    n_in, n_out = w_arr.shape

                    next2 = nodes[i + 2].op_type if i + 2 < len(nodes) else None
                    if next2 == "Relu":
                        fn  = "ml_dense_relu"
                        inc2 = 3
                    elif next2 == "Softmax":
                        fn   = "ml_dense_softmax"
                        inc2 = 3
                    else:
                        fn   = "ml_dense_linear"
                        inc2 = 2

                    calls.append(f"    /* Dense({n_out}) [{fn}] */")
                    calls.append(
                        f"    {fn}(in_ptr, {n_in}, "
                        f"{self._wref(w_name)}, {self._wref(b_name)}, {n_out}, out_ptr);")
                    swap()
                    i += inc2
                else:
                    calls.append(f"    /* WARNING: MatMul without Add — skipped */")
                    i += 1

            elif op in ("Add", "Softmax"):
                # consumed by MatMul handler; shouldn't arrive standalone
                calls.append(f"    /* {op} — already consumed above */")
                i += 1

            else:
                calls.append(f"    /* WARNING: unsupported op '{op}' — skipped */")
                i += 1

        calls.append("")
        calls.append("    /* Copy scores to output, return argmax */")
        calls.append(f"    memcpy(scores, in_ptr, {n_classes} * sizeof(float));")
        calls.append("    int best = 0;")
        calls.append(f"    for (int j = 1; j < {n_classes}; j++)")
        calls.append("        if (in_ptr[j] > in_ptr[best]) best = j;")
        calls.append("    return best;")

        forward_body = "\n".join(calls)

        header = f"""/* model.h — generated by deploy.py (ONNX path)
   Model: {model_name}   Classes: {n_classes}   Arena: {arena_kb:.1f} KB  */
#pragma once
#include <stdint.h>
#define MODEL_ARENA_BYTES  {peak_bytes}
#define MODEL_N_CLASSES    {n_classes}
#ifdef __cplusplus
extern "C" {{
#endif
void model_init(void);
int model_run(const float *input, float *scores, float *arena);
#ifdef __cplusplus
}}
#endif
"""

        source = f"""/* model.c — generated by deploy.py (ONNX path) */
#include "model.h"
#include "weights.h"
#include <string.h>
#include <math.h>

{_C_PRIMITIVES_NCHW}

void model_init(void) {{}}

int model_run(const float *input, float *scores, float *arena) {{
{forward_body}
}}
"""
        (inc / "model.h").write_text(header)
        (src / "model.c").write_text(source)

    # ------------------------------------------------------------------
    # Feature data, feature extraction, audio, main, platformio.ini
    # (identical to ModelToC — delegate via import)
    # ------------------------------------------------------------------

    def _gen_feat_data(self, src: Path, inc: Path) -> None:
        from .model_to_c import ModelToC
        _stub = object.__new__(ModelToC)
        _stub.fp = self.fp
        ModelToC._gen_feat_data(_stub, src, inc)

    def _gen_features(self, src: Path, inc: Path) -> None:
        from .model_to_c import _FEATURES_H, _FEATURES_C
        n_frames  = self.fp.get("n_frames") or (
                    1 + int(round(self.fp["duration"] * self.fp["sample_rate"])) // self.fp["hop_length"])
        n_samples = (n_frames - 1) * self.fp["hop_length"]
        header = _FEATURES_H.format(
            sample_rate = self.fp["sample_rate"],
            n_fft       = self.fp["n_fft"],
            hop_length  = self.fp["hop_length"],
            n_mels      = self.fp["n_mels"],
            n_samples   = n_samples,
            n_frames    = n_frames,
        )
        (inc / "features.h").write_text(header)
        (src / "features.c").write_text(_FEATURES_C)

    def _gen_audio(self, src: Path, inc: Path) -> None:
        from .model_to_c import ModelToC
        _stub = object.__new__(ModelToC)
        _stub.board_cfg  = self.board_cfg
        _stub.board_name = self.board_name
        _stub.fp         = self.fp
        ModelToC._gen_audio(_stub, src, inc)

    def _gen_main(self, src: Path, peak_bytes: int) -> None:
        from .model_to_c import ModelToC, _MAIN_CPP
        n_classes  = len(self.labels)
        arena_kb   = peak_bytes / 1024
        label_list = ", ".join(f'"{l}"' for l in self.labels)
        txt = _MAIN_CPP.format(
            board      = self.board_name,
            model_name = self.onnx_path.stem,
            n_classes  = n_classes,
            arena_kb   = arena_kb,
            label_list = label_list,
        )
        (src / "main.cpp").write_text(txt)

    def _gen_platformio_ini(self) -> None:
        from .model_to_c import ModelToC
        _stub = object.__new__(ModelToC)
        _stub.board_cfg  = self.board_cfg
        _stub.board_name = self.board_name
        _stub.output_dir = self.output_dir
        ModelToC._gen_platformio_ini(_stub)
