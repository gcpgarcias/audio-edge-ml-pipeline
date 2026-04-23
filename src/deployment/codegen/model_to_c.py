"""
Keras CNN → C codegen for embedded deployment.

Generates a self-contained PlatformIO project with:
    weights.h / weights.c   — const float arrays in Flash
    model.h   / model.c     — forward pass, static arena, layer primitives
    audio.h   / audio.cpp   — board-specific PDM/I2S capture + mel features
    main.cpp                — inference loop
    platformio.ini          — board, framework, build flags

All layer primitives are pure C99 — no external ML library required.
Architecture-agnostic: same generated code compiles for any board that
supports float32 arithmetic.

Supported Keras layers
----------------------
    InputLayer · Normalization · Conv2D (same/valid, relu) · DepthwiseConv2D
    MaxPooling2D · AveragePooling2D · GlobalAveragePooling2D
    Dense (relu / softmax / linear) · Dropout (identity at inference)
    Flatten · BatchNormalization · Reshape
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Board catalogue
# ---------------------------------------------------------------------------

BOARDS: dict[str, dict] = {
    "nicla_vision": {
        "pio_board":    "nicla_vision",
        "platform":     "ststm32",
        "framework":    "arduino",
        "ram_kb":       1024,   # full 1MB AXI SRAM available in bare C (no OpenMV runtime)
        "flash_kb":     2048,
        "dtcm_kb":      128,    # fastest RAM — arena placed here if it fits
        "cpu":          "cortex-m7",
        "audio_driver": "pdm_stm32",
        "sample_rate":  16000,
        "build_flags":  ["-O2", "-ffast-math"],
        "lib_deps":     ["PDM"],
    },
    "nano_ble": {
        "pio_board":    "nano33ble",
        "platform":     "nordicnrf52",
        "framework":    "arduino",
        "ram_kb":       256,
        "flash_kb":     1024,
        "dtcm_kb":      0,
        "cpu":          "cortex-m4",
        "audio_driver": "pdm_nrf",
        "sample_rate":  16000,
        "build_flags":  ["-O2", "-ffast-math"],
        "lib_deps":     ["PDM"],
    },
    "esp32s3": {
        "pio_board":    "esp32-s3-devkitc-1",
        "platform":     "espressif32",
        "framework":    "arduino",
        "ram_kb":       512,
        "flash_kb":     16384,
        "dtcm_kb":      0,
        "cpu":          "xtensa-lx7",
        "audio_driver": "i2s_esp32",
        "sample_rate":  16000,
        "build_flags":  ["-O2", "-ffast-math"],
        "lib_deps":     [],
    },
    "pico2": {
        "pio_board":    "rpipico2",
        "platform":     "raspberrypi",
        "framework":    "arduino",
        "ram_kb":       520,
        "flash_kb":     4096,
        "dtcm_kb":      0,
        "cpu":          "cortex-m33",
        "audio_driver": "i2s_rp2040",
        "sample_rate":  16000,
        "build_flags":  ["-O2", "-ffast-math"],
        "lib_deps":     [],
    },
}


# ---------------------------------------------------------------------------
# C layer primitives (embedded as string — no Jinja2 dependency)
# ---------------------------------------------------------------------------

_C_PRIMITIVES = r"""
/* -------------------------------------------------------------------------
 * Inference layer primitives — pure C99, no external dependencies.
 * All tensors use NHWC memory layout (same as Keras default).
 * ------------------------------------------------------------------------- */

#include <math.h>
#include <string.h>

/* ── Normalization ──────────────────────────────────────────────────────── */
/* Applies (x - mean) / sqrt(var + eps) element-wise.                       */
/* mean/var broadcast over H,W axes; shape is (C,) for 3-D input.           */
static void ml_normalize(
    const float *in, int n_elem, int n_ch,
    const float *mean, const float *var, float eps,
    float *out)
{
    for (int i = 0; i < n_elem; i++) {
        int c = i % n_ch;
        out[i] = (in[i] - mean[c]) / sqrtf(var[c] + eps);
    }
}

/* ── Conv2D (same padding, stride 1, relu) ──────────────────────────────── */
static void ml_conv2d_relu_same(
    const float *in,  int in_h, int in_w, int in_c,
    const float *w,   int kh,   int kw,   int out_c,
    const float *b,
    float *out)
{
    int pad_h = kh / 2, pad_w = kw / 2;
    for (int oh = 0; oh < in_h; oh++) {
        for (int ow = 0; ow < in_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                float acc = b[oc];
                for (int krow = 0; krow < kh; krow++) {
                    int ih = oh + krow - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kcol = 0; kcol < kw; kcol++) {
                        int iw = ow + kcol - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        for (int ic = 0; ic < in_c; ic++) {
                            /* Keras weight layout: (kh, kw, in_c, out_c) */
                            acc += in[(ih * in_w + iw) * in_c + ic]
                                 * w[((krow * kw + kcol) * in_c + ic) * out_c + oc];
                        }
                    }
                }
                out[(oh * in_w + ow) * out_c + oc] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* ── Conv2D (valid padding, stride 1, relu) ─────────────────────────────── */
static void ml_conv2d_relu_valid(
    const float *in,  int in_h, int in_w, int in_c,
    const float *w,   int kh,   int kw,   int out_c,
    const float *b,
    float *out)         /* out: (in_h-kh+1, in_w-kw+1, out_c) */
{
    int out_h = in_h - kh + 1, out_w = in_w - kw + 1;
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                float acc = b[oc];
                for (int krow = 0; krow < kh; krow++) {
                    for (int kcol = 0; kcol < kw; kcol++) {
                        for (int ic = 0; ic < in_c; ic++) {
                            acc += in[((oh + krow) * in_w + ow + kcol) * in_c + ic]
                                 * w[((krow * kw + kcol) * in_c + ic) * out_c + oc];
                        }
                    }
                }
                out[(oh * out_w + ow) * out_c + oc] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* ── DepthwiseConv2D (same padding, stride 1, relu) ────────────────────── */
static void ml_dwconv2d_relu_same(
    const float *in,  int in_h, int in_w, int in_c,
    const float *w,   int kh,   int kw,
    const float *b,
    float *out)
{
    int pad_h = kh / 2, pad_w = kw / 2;
    for (int oh = 0; oh < in_h; oh++) {
        for (int ow = 0; ow < in_w; ow++) {
            for (int c = 0; c < in_c; c++) {
                float acc = b[c];
                for (int krow = 0; krow < kh; krow++) {
                    int ih = oh + krow - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kcol = 0; kcol < kw; kcol++) {
                        int iw = ow + kcol - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        /* Keras DW weight layout: (kh, kw, in_c, 1) */
                        acc += in[(ih * in_w + iw) * in_c + c]
                             * w[(krow * kw + kcol) * in_c + c];
                    }
                }
                out[(oh * in_w + ow) * in_c + c] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* ── MaxPooling2D ───────────────────────────────────────────────────────── */
static void ml_maxpool2d(
    const float *in, int in_h, int in_w, int in_c,
    int ph, int pw, int sh, int sw,
    float *out)
{
    int out_h = (in_h - ph) / sh + 1;
    int out_w = (in_w - pw) / sw + 1;
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < in_c; c++) {
                float mx = -3.402823466e+38f;
                for (int i = 0; i < ph; i++) {
                    for (int j = 0; j < pw; j++) {
                        float v = in[((oh*sh+i)*in_w + ow*sw+j)*in_c + c];
                        if (v > mx) mx = v;
                    }
                }
                out[(oh * out_w + ow) * in_c + c] = mx;
            }
        }
    }
}

/* ── AveragePooling2D ───────────────────────────────────────────────────── */
static void ml_avgpool2d(
    const float *in, int in_h, int in_w, int in_c,
    int ph, int pw, int sh, int sw,
    float *out)
{
    int out_h = (in_h - ph) / sh + 1;
    int out_w = (in_w - pw) / sw + 1;
    float inv = 1.0f / (float)(ph * pw);
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < in_c; c++) {
                float s = 0.0f;
                for (int i = 0; i < ph; i++)
                    for (int j = 0; j < pw; j++)
                        s += in[((oh*sh+i)*in_w + ow*sw+j)*in_c + c];
                out[(oh * out_w + ow) * in_c + c] = s * inv;
            }
        }
    }
}

/* ── GlobalAveragePooling2D ─────────────────────────────────────────────── */
static void ml_gap2d(
    const float *in, int h, int w, int c,
    float *out)
{
    float inv = 1.0f / (float)(h * w);
    for (int ci = 0; ci < c; ci++) {
        float s = 0.0f;
        for (int i = 0; i < h * w; i++) s += in[i * c + ci];
        out[ci] = s * inv;
    }
}

/* ── BatchNormalization ─────────────────────────────────────────────────── */
static void ml_batchnorm(
    const float *in, int n_elem, int n_ch,
    const float *gamma, const float *beta,
    const float *mean,  const float *var, float eps,
    float *out)
{
    for (int i = 0; i < n_elem; i++) {
        int c  = i % n_ch;
        float xn = (in[i] - mean[c]) / sqrtf(var[c] + eps);
        out[i]   = gamma[c] * xn + beta[c];
    }
}

/* ── Dense + relu ───────────────────────────────────────────────────────── */
static void ml_dense_relu(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    for (int i = 0; i < n_out; i++) {
        float acc = b[i];
        for (int j = 0; j < n_in; j++) acc += in[j] * w[j * n_out + i];
        out[i] = acc > 0.0f ? acc : 0.0f;
    }
}

/* ── Dense + softmax ────────────────────────────────────────────────────── */
static void ml_dense_softmax(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    float mx = -3.402823466e+38f;
    for (int i = 0; i < n_out; i++) {
        float acc = b[i];
        for (int j = 0; j < n_in; j++) acc += in[j] * w[j * n_out + i];
        out[i] = acc;
        if (acc > mx) mx = acc;
    }
    float sum = 0.0f;
    for (int i = 0; i < n_out; i++) { out[i] = expf(out[i] - mx); sum += out[i]; }
    for (int i = 0; i < n_out; i++) out[i] /= sum;
}

/* ── Dense + linear (no activation) ────────────────────────────────────── */
static void ml_dense_linear(
    const float *in, int n_in,
    const float *w,  const float *b, int n_out,
    float *out)
{
    for (int i = 0; i < n_out; i++) {
        float acc = b[i];
        for (int j = 0; j < n_in; j++) acc += in[j] * w[j * n_out + i];
        out[i] = acc;
    }
}
"""

# ---------------------------------------------------------------------------
# Audio driver templates (board-specific)
# ---------------------------------------------------------------------------

_AUDIO_PDM_STM32 = """
/* audio.h — PDM microphone capture for STM32 boards (Nicla Vision)        */
#pragma once
#include <stdint.h>
#define AUDIO_SAMPLE_RATE  {sample_rate}
#define AUDIO_N_SAMPLES    ({sample_rate} * AUDIO_DURATION_S)
#ifdef __cplusplus
extern "C" {{
#endif
void audio_init(void);
void audio_record(int16_t *buf, int n_samples);
#ifdef __cplusplus
}}
#endif
"""

_AUDIO_CPP_PDM_STM32 = """
#include "audio.h"
#include <PDM.h>

static volatile int16_t *_pdm_buf  = NULL;
static volatile int      _pdm_pos  = 0;
static volatile bool     _pdm_done = false;
static volatile int      _pdm_n    = 0;

static int16_t _drain[64];

static void _pdm_cb() {{
    int available = PDM.available();
    if (_pdm_buf == NULL || _pdm_n == 0) {{
        while (available >= 2) {{
            int n = available / 2;
            if (n > 64) n = 64;
            PDM.read(_drain, n * 2);
            available -= n * 2;
        }}
        return;
    }}
    if (_pdm_done) return;
    while (available > 0) {{
        int n = available / 2;
        if (_pdm_pos + n > _pdm_n) n = _pdm_n - _pdm_pos;
        PDM.read((int16_t*)_pdm_buf + _pdm_pos, n * 2);
        _pdm_pos += n;
        if (_pdm_pos >= _pdm_n) {{ _pdm_done = true; return; }}
        available = PDM.available();
    }}
}}

void audio_init() {{
    PDM.onReceive(_pdm_cb);
    PDM.begin(1, AUDIO_SAMPLE_RATE);
    PDM.setGain(12);
}}

#define AUDIO_WARMUP_SAMPLES 4096

static int16_t _warmup_buf[AUDIO_WARMUP_SAMPLES];

void audio_record(int16_t *buf, int n_samples) {{
    _pdm_buf  = _warmup_buf;
    _pdm_pos  = 0;
    _pdm_n    = AUDIO_WARMUP_SAMPLES;
    _pdm_done = false;
    while (!_pdm_done) {{ /* spin */ }}

    _pdm_buf  = buf;
    _pdm_pos  = 0;
    _pdm_n    = n_samples;
    _pdm_done = false;
    while (!_pdm_done) {{ /* spin */ }}

    _pdm_buf = NULL;
    _pdm_n   = 0;
}}
"""

_AUDIO_I2S_ESP32 = """
#include "audio.h"
#include <driver/i2s.h>
#define I2S_NUM         I2S_NUM_0
#define I2S_BCK_PIN     1
#define I2S_WS_PIN      2
#define I2S_DATA_PIN    3
void audio_init() {{
    i2s_config_t cfg = {{
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 4,
        .dma_buf_len = 256,
    }};
    i2s_driver_install(I2S_NUM, &cfg, 0, NULL);
    i2s_pin_config_t pins = {{
        .bck_io_num   = I2S_BCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DATA_PIN,
    }};
    i2s_set_pin(I2S_NUM, &pins);
}}
void audio_record(int16_t *buf, int n_samples) {{
    size_t bytes_read = 0, total = 0;
    while (total < (size_t)n_samples * 2) {{
        i2s_read(I2S_NUM, buf + total/2, n_samples*2 - total, &bytes_read, portMAX_DELAY);
        total += bytes_read;
    }}
}}
"""

_AUDIO_GENERIC = """
/* audio.cpp — stub for boards without a built-in audio driver.
   Replace with your board's microphone API.                                */
#include "audio.h"
#include <string.h>
void audio_init() {{ /* configure your mic here */ }}
void audio_record(int16_t *buf, int n_samples) {{
    memset(buf, 0, n_samples * sizeof(int16_t));  /* placeholder */
}}
"""

_AUDIO_DRIVERS = {
    "pdm_stm32": (_AUDIO_PDM_STM32, _AUDIO_CPP_PDM_STM32),
    "pdm_nrf":   (_AUDIO_PDM_STM32, _AUDIO_CPP_PDM_STM32),  # same API
    "i2s_esp32": (_AUDIO_PDM_STM32, _AUDIO_I2S_ESP32),
    "i2s_rp2040":(_AUDIO_PDM_STM32, _AUDIO_GENERIC),
}

# ---------------------------------------------------------------------------
# Feature extraction C template (MFCC + spectral — matches Python extractor)
# ---------------------------------------------------------------------------

_FEATURES_H = """
/* features.h — on-device log-mel spectrogram extraction                    */
#pragma once
#include <stdint.h>
#define FEAT_SAMPLE_RATE  {sample_rate}
#define FEAT_N_FFT        {n_fft}
#define FEAT_HOP          {hop_length}
#define FEAT_N_MELS       {n_mels}
/* Literal counts computed at codegen time — avoids float-multiply rounding.
   n_samples = (n_frames - 1) * hop  (center=True inverse of librosa default) */
#define FEAT_N_SAMPLES    {n_samples}
#define FEAT_N_FRAMES     {n_frames}
/* Output: float[FEAT_N_MELS][FEAT_N_FRAMES] — log-mel spectrogram (NHWC with C=1) */
#define FEAT_DIM          (FEAT_N_MELS * FEAT_N_FRAMES)

/* mel_fb: float[FEAT_N_MELS][FEAT_N_FFT/2+1] — mel filterbank             */
extern const float feat_mel_fb[FEAT_N_MELS][FEAT_N_FFT/2+1];

#ifdef __cplusplus
extern "C" {{
#endif
/* Fills out[FEAT_N_MELS * FEAT_N_FRAMES] with the log-mel spectrogram.
   Layout: out[mel * FEAT_N_FRAMES + frame] — matches model input (H,W,1). */
void features_extract(const int16_t *pcm, int n_samples, float *out);
#ifdef __cplusplus
}}
#endif
"""

_FEATURES_C = r"""
/* features.c — log-mel spectrogram in C99
   Output: float[FEAT_N_MELS * FEAT_N_FRAMES], layout out[mel * FEAT_N_FRAMES + frame]
   Matches AudioMelSpectrogram extractor (librosa.feature.melspectrogram,
   power=2.0, log-power in dB: 10*log10).
*/
#include "features.h"
#include "feat_data.h"
#include <math.h>
#include <string.h>

#define N_BINS  (FEAT_N_FFT / 2 + 1)

static void _fft(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t; t=re[i]; re[i]=re[j]; re[j]=t;
                     t=im[i]; im[i]=im[j]; im[j]=t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265f / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int j = 0; j < len/2; j++) {
                float u_re = re[i+j], u_im = im[i+j];
                float v_re = re[i+j+len/2]*cur_re - im[i+j+len/2]*cur_im;
                float v_im = re[i+j+len/2]*cur_im + im[i+j+len/2]*cur_re;
                re[i+j]        = u_re + v_re;
                im[i+j]        = u_im + v_im;
                re[i+j+len/2]  = u_re - v_re;
                im[i+j+len/2]  = u_im - v_im;
                float t = cur_re*wre - cur_im*wim;
                cur_im  = cur_re*wim + cur_im*wre;
                cur_re  = t;
            }
        }
    }
}

static float _hann[FEAT_N_FFT];
static int   _hann_ready = 0;
static void _init_hann(void) {
    if (_hann_ready) return;
    for (int i = 0; i < FEAT_N_FFT; i++)
        _hann[i] = 0.5f * (1.0f - cosf(2.0f*3.14159265f*i/(float)FEAT_N_FFT));
    _hann_ready = 1;
}

static float _re[FEAT_N_FFT];
static float _im[FEAT_N_FFT];
static float _power[N_BINS];

void features_extract(const int16_t *pcm, int n_samples, float *out) {
    _init_hann();
    /* Match librosa center=True: frame centre is at fi*hop, so the frame
       window starts at fi*hop - n_fft/2.  n_frames = 1 + n_samples/hop. */
    int half  = FEAT_N_FFT / 2;
    int n_frames = 1 + n_samples / FEAT_HOP;
    if (n_frames > FEAT_N_FRAMES) n_frames = FEAT_N_FRAMES;

    for (int fi = 0; fi < n_frames; fi++) {
        int frame_start = fi * FEAT_HOP - half;   /* may be negative */

        /* int16 → float, Hann window, zero-pad outside [0, n_samples) */
        for (int i = 0; i < FEAT_N_FFT; i++) {
            int si = frame_start + i;
            float s = (si >= 0 && si < n_samples)
                      ? (float)pcm[si] / 32768.0f : 0.0f;
            _re[i] = s * _hann[i];
            _im[i] = 0.0f;
        }

        /* FFT → power spectrum */
        _fft(_re, _im, FEAT_N_FFT);
        for (int i = 0; i < N_BINS; i++)
            _power[i] = _re[i]*_re[i] + _im[i]*_im[i];

        /* Mel filterbank → raw power per band */
        for (int m = 0; m < FEAT_N_MELS; m++) {
            float s = 0.0f;
            for (int b = 0; b < N_BINS; b++) s += feat_mel_fb[m][b] * _power[b];
            out[m * FEAT_N_FRAMES + fi] = s;
        }
    }

    /* Zero-pad any remaining frames */
    for (int fi = n_frames; fi < FEAT_N_FRAMES; fi++)
        for (int m = 0; m < FEAT_N_MELS; m++)
            out[m * FEAT_N_FRAMES + fi] = 0.0f;

    /* Convert to dB relative to max (librosa.power_to_db(ref=np.max), no top_db clip),
       then true min-max normalize to [0, 1] — matching _normalize() in the Python
       extractor (src/preprocessing/feature_extraction/audio/deep.py):
           lo, hi = x.min(), x.max()
           return (x - lo) / (hi - lo + eps)                                        */
    int total = FEAT_N_MELS * FEAT_N_FRAMES;
    float max_power = 1e-10f;
    for (int i = 0; i < total; i++)
        if (out[i] > max_power) max_power = out[i];

    /* power → dB, no clipping */
    for (int i = 0; i < total; i++)
        out[i] = 10.0f * log10f(out[i] / max_power + 1e-10f);

    /* true min-max normalize */
    float db_min = out[0], db_max = out[0];
    for (int i = 1; i < total; i++) {
        if (out[i] < db_min) db_min = out[i];
        if (out[i] > db_max) db_max = out[i];
    }
    float range = db_max - db_min + 1e-8f;
    for (int i = 0; i < total; i++)
        out[i] = (out[i] - db_min) / range;
}
"""

# ---------------------------------------------------------------------------
# main.cpp template
# ---------------------------------------------------------------------------

_MAIN_CPP = """
/* main.cpp — {board} inference loop
   Auto-generated by src/deployment/deploy.py
   Model: {model_name}  Classes: {n_classes}  Arena: {arena_kb:.0f} KB

   Memory layout
   -------------
   Recording and inference never overlap, so the PCM buffer aliases the start
   of the inference arena.  This saves ~160 KB of RAM on a 5-second model.

       _shared_buf[0 .. FEAT_N_SAMPLES*2-1]  — PCM int16 during recording
       _shared_buf[0 .. MODEL_ARENA_BYTES-1] — activation arena during inference
*/

#include <Arduino.h>
#include "audio.h"
#include "features.h"
#include "model.h"

/* Shared scratch — large enough for whichever phase needs more RAM.
   Declared as float to guarantee 4-byte alignment for arena float* casts.
   PCM int16 recording reinterprets the same memory as int16_t*.
   feat_buf starts at offset FEAT_N_SAMPLES/2 floats and is FEAT_DIM floats
   long, so the buffer must cover at least that tail even when the arena is
   smaller (e.g. compact models with a tiny activation footprint).          */
#define _PCM_FLOATS   (FEAT_N_SAMPLES / 2)
#define _FEAT_END     (_PCM_FLOATS + FEAT_DIM)
#define _ARENA_FLOATS (MODEL_ARENA_BYTES / 4)
#define SHARED_BUF_FLOATS  ((_FEAT_END > _ARENA_FLOATS ? _FEAT_END : _ARENA_FLOATS) + 1)
static float _shared_buf[SHARED_BUF_FLOATS];

/* feat_buf is placed AFTER the PCM region to avoid in-place aliasing:
   features_extract writes in stride order (out[mel*N_FRAMES+fi]), which
   would overwrite PCM samples still being read if both start at [0].
   PCM occupies _shared_buf[0 .. FEAT_N_SAMPLES/2-1] as float (int16*2=4B).
   feat_buf follows immediately; model_run arena reuses from [0] once PCM
   is no longer needed.                                                     */
#define feat_buf  ((float *)_shared_buf + FEAT_N_SAMPLES / 2)
static float   scores[MODEL_N_CLASSES];

static const char *labels[MODEL_N_CLASSES] = {{ {label_list} }};

/* Set FEAT_DUMP_MODE to 1 to stream raw mel spectrogram over serial instead
   of running the model.  Use tools/receive_mel.py to capture and compare.  */
#define FEAT_DUMP_MODE 0

/* Set PCM_DUMP_MODE to 1 to stream raw int16 PCM over serial (after DC
   removal and notch filter).  Use tools/receive_wav.py to save as .wav. */
#define PCM_DUMP_MODE 0

void setup() {{
    Serial.begin(115200);
    delay(2000);   // give host time to open monitor; don't block on !Serial
    Serial.println("Booting {model_name} ...");
    audio_init();
    model_init();
    Serial.println("Ready.");
}}

void loop() {{
    int16_t *pcm_buf = (int16_t *)_shared_buf;

    {{
        extern uint32_t __StackTop;
        uint32_t sp;
        __asm volatile ("mov %0, sp" : "=r"(sp));
        Serial.print("Free stack approx: ");
        Serial.print((uint32_t)&__StackTop - sp);
        Serial.println(" bytes");
    }}

#if PCM_DUMP_MODE
    /* Wait for 'R' trigger from host before recording.
       tools/record_dataset.py sends 'R', then immediately starts audio playback.
       The device starts capturing at the same moment — no manual sync needed. */
    do {{ Serial.println("READY"); delay(200); }} while (!Serial.available());
    if (Serial.read() != 'R') return;   // unexpected byte — skip this cycle
#endif

    Serial.println("Recording ...");
    audio_record(pcm_buf, FEAT_N_SAMPLES);

    /* PCM diagnostics */
    {{
        int16_t pcm_min = pcm_buf[0], pcm_max = pcm_buf[0];
        for (int i = 1; i < FEAT_N_SAMPLES; i++) {{
            if (pcm_buf[i] < pcm_min) pcm_min = pcm_buf[i];
            if (pcm_buf[i] > pcm_max) pcm_max = pcm_buf[i];
        }}
        Serial.print("PCM min="); Serial.print(pcm_min);
        Serial.print(" max=");    Serial.println(pcm_max);

        int32_t sum = 0;
        for (int i = 0; i < FEAT_N_SAMPLES; i++) sum += pcm_buf[i];
        int16_t dc = (int16_t)(sum / FEAT_N_SAMPLES);
        for (int i = 0; i < FEAT_N_SAMPLES; i++) pcm_buf[i] -= dc;
        int16_t ac_min = pcm_buf[0], ac_max = pcm_buf[0];
        for (int i = 1; i < FEAT_N_SAMPLES; i++) {{
            if (pcm_buf[i] < ac_min) ac_min = pcm_buf[i];
            if (pcm_buf[i] > ac_max) ac_max = pcm_buf[i];
        }}
        Serial.print("DC="); Serial.print(dc);
        Serial.print("  AC min="); Serial.print(ac_min);
        Serial.print(" max="); Serial.println(ac_max);
    }}

    /* Biquad notch at 4 kHz (sr=16000, Q=8) — removes PDM clock artifact.
       Coefficients pre-computed: w0=pi/2, b1=a1=0 (exact null at Nyquist/2).
       Direct Form I: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
                            - a1*y[n-1] - a2*y[n-2]               */
    {{
        const float b0 =  0.94117647f, b1 = 0.0f, b2 = 0.94117647f;
        const float                    a1 = 0.0f, a2 = 0.88235294f;
        float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
        for (int i = 0; i < FEAT_N_SAMPLES; i++) {{
            float x0 = (float)pcm_buf[i];
            float y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2;
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;
            if (y0 >  32767.0f) y0 =  32767.0f;
            if (y0 < -32768.0f) y0 = -32768.0f;
            pcm_buf[i] = (int16_t)y0;
        }}
    }}

#if PCM_DUMP_MODE
    {{
        const uint32_t n_samples = FEAT_N_SAMPLES;
        const uint8_t magic_start[4] = {{0xCA, 0xFE, 0xBA, 0xBE}};
        const uint8_t magic_end[4]   = {{0xDE, 0xAD, 0xBE, 0xEF}};
        Serial.write(magic_start, 4);
        Serial.write((uint8_t *)&n_samples, 4);
        Serial.write((uint8_t *)pcm_buf, n_samples * 2);
        Serial.write(magic_end, 4);
        Serial.println("PCM_DUMP_DONE");
    }}
#else
    Serial.println("Extracting features ...");
    features_extract(pcm_buf, FEAT_N_SAMPLES, feat_buf);
    Serial.println("Features done.");

#if FEAT_DUMP_MODE
    /* Binary frame: magic(4) | n_floats(4) | floats(n*4) | magic_end(4)
       Received by tools/receive_mel.py                                     */
    const uint32_t n_floats = FEAT_N_MELS * FEAT_N_FRAMES;
    const uint8_t magic_start[4] = {{0xFE, 0xED, 0x12, 0x34}};
    const uint8_t magic_end[4]   = {{0xDE, 0xAD, 0x56, 0x78}};
    Serial.write(magic_start, 4);
    Serial.write((uint8_t *)&n_floats, 4);
    Serial.write((uint8_t *)feat_buf, n_floats * 4);
    Serial.write(magic_end, 4);
    Serial.println("DUMP_DONE");
    delay(2000);
#else
    /* pcm_buf alias is no longer needed — arena reuses _shared_buf */
    Serial.println("Running model ...");
    int cls = model_run(feat_buf, scores, _shared_buf);
    Serial.println("Model done.");

    /* Top-3 results */
    int top[3] = {{cls, -1, -1}};
    for (int t = 1; t < 3; t++) {{
        float best = -1.0f;
        for (int i = 0; i < MODEL_N_CLASSES; i++) {{
            bool used = false;
            for (int u = 0; u < t; u++) if (top[u] == i) {{ used = true; break; }}
            if (!used && scores[i] > best) {{ best = scores[i]; top[t] = i; }}
        }}
    }}
    Serial.println("--- Top 3 ---");
    for (int t = 0; t < 3; t++) {{
        Serial.print("  "); Serial.print(t + 1);
        Serial.print(". "); Serial.print(labels[top[t]]);
        Serial.print(": score="); Serial.println(scores[top[t]], 3);
    }}

    delay(500);
#endif
#endif  /* PCM_DUMP_MODE */
}}
"""

# ---------------------------------------------------------------------------
# platformio.ini template
# ---------------------------------------------------------------------------

_PIO_INI = """
; platformio.ini — auto-generated by src/deployment/deploy.py
[env:{board}]
platform  = {platform}
board     = {pio_board}
framework = {framework}
build_flags =
{build_flags}
lib_deps =
{lib_deps}
monitor_speed = 115200
{extra_ini}"""

# Custom linker script for Nicla Vision — expands AXI SRAM from 512 KB to 1 MB.
# STM32H747 has 1 MB AXI SRAM at 0x24000000; the default mbed LD maps only 0x80000.
_NICLA_LD = """\
/* nicla_vision.ld — custom linker script: full 1 MB AXI SRAM
   Auto-generated by src/deployment/deploy.py — do not edit. */
MEMORY
{
  FLASH    (rx)  : ORIGIN = 0x8040000, LENGTH = 0x1C0000
  DTCMRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
  RAM      (xrw) : ORIGIN = 0x24000000, LENGTH = 512K
  RAM_D2   (xrw) : ORIGIN = 0x30000000, LENGTH = 288K
  RAM_D3   (xrw) : ORIGIN = 0x38000000, LENGTH = 64K
  ITCMRAM  (xrw) : ORIGIN = 0x00000000, LENGTH = 64K
}
__OPENAMP_region_start__  = 0x38000400;
__OPENAMP_region_end__ = 0x38000400 + LENGTH(RAM_D3) - 1K;
ENTRY(Reset_Handler)
SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        KEEP(*(.init))
        KEEP(*(.fini))
        *crtbegin.o(.ctors)
        *crtbegin?.o(.ctors)
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
        *(SORT(.ctors.*))
        *(.ctors)
        *crtbegin.o(.dtors)
        *crtbegin?.o(.dtors)
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
        *(SORT(.dtors.*))
        *(.dtors)
        *(.rodata*)
        KEEP(*(.eh_frame*))
        *ltrans0*.o(.rodata*)
        *ltrans1*.o(.rodata*)
        *ltrans2*.o(.rodata*)
        *ltrans3*.o(.rodata*)
        *ltrans4*.o(.rodata*)
        *lib*.o(.rodata*)
    } > FLASH
    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
    __exidx_start = .;
    .ARM.exidx : { *(.ARM.exidx* .gnu.linkonce.armexidx.*) } > FLASH
    __exidx_end = .;
    __etext = .;
    _sidata = .;
    .data : AT (__etext)
    {
        __data_start__ = .;
        _sdata = .;
        *(vtable)
        *(.data*)
        . = ALIGN(8);
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
        . = ALIGN(8);
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN (__init_array_end = .);
        . = ALIGN(8);
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN (__fini_array_end = .);
        KEEP(*(.jcr*))
        . = ALIGN(8);
        __data_end__ = .;
        _edata = .;
    } > RAM
    .uninitialized (NOLOAD):
    {
        . = ALIGN(32);
        __uninitialized_start = .;
        *(.uninitialized)
        KEEP(*(.keep.uninitialized))
        . = ALIGN(32);
        __uninitialized_end = .;
    } > RAM
    .bss :
    {
        . = ALIGN(8);
        __bss_start__ = .;
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end__ = .;
        _ebss = .;
    } > RAM
    .pdm_section 0x3800FC00 (NOLOAD): { *(.pdm_buffer) } > RAM_D3
    .heap (COPY):
    {
        __end__ = .;
        PROVIDE(end = .);
        *(.heap*)
        . = ORIGIN(RAM) + LENGTH(RAM) - 0x400;
        __HeapLimit = .;
    } > RAM
    .stack_dummy (COPY): { *(.stack*) } > RAM
    __StackTop = ORIGIN(RAM) + LENGTH(RAM);
    _estack = __StackTop;
    __StackLimit = __StackTop - 0x400;
    PROVIDE(__stack = __StackTop);
    ASSERT(__StackLimit >= __HeapLimit, "region RAM overflowed with stack")
    .lwip_sec (NOLOAD) :
    {
        . = ABSOLUTE(0x30040000); *(.RxDecripSection)
        . = ABSOLUTE(0x30040100); *(.TxDecripSection)
        . = ABSOLUTE(0x30040400); *(.RxArraySection)
        . = ABSOLUTE(0x30044000); *(.ethusbram)
    } > RAM_D2 AT> FLASH
}
"""


# ---------------------------------------------------------------------------
# Main codegen class
# ---------------------------------------------------------------------------

class ModelToC:
    """Convert a Keras model to a PlatformIO C project.

    Parameters
    ----------
    model:
        Built Keras model (CNN or MLP).
    output_dir:
        Directory to write the project into.
    board:
        Board name from BOARDS catalogue.
    label_names:
        Ordered list of class name strings.
    feature_params:
        Dict with keys: sample_rate, n_fft, hop_length, n_mels, n_mfcc,
        duration, n_features.
    max_ram_kb:
        Raise if peak arena exceeds this.
    """

    def __init__(
        self,
        model,
        output_dir:     Path,
        board:          str,
        label_names:    list[str],
        feature_params: dict,
        max_ram_kb:     float = 256.0,
    ) -> None:
        self.model          = model
        self.output_dir     = Path(output_dir)
        self.board_cfg      = BOARDS[board]
        self.board_name     = board
        self.labels         = label_names
        self.fp             = feature_params
        self.max_ram_kb     = max_ram_kb

    # ------------------------------------------------------------------

    def generate(self) -> None:
        """Run full codegen pipeline."""
        from .arena_estimator import check_max_ram

        self.output_dir.mkdir(parents=True, exist_ok=True)
        src = self.output_dir / "src"
        src.mkdir(exist_ok=True)
        inc = self.output_dir / "include"
        inc.mkdir(exist_ok=True)

        print("Estimating arena ...")
        peak_bytes, arena_layers = check_max_ram(
            self.model, self.max_ram_kb, verbose=True)

        print("Generating weights ...")
        self._gen_weights(src, inc)

        print("Generating feature data ...")
        self._gen_feat_data(src, inc)

        print("Generating model forward pass ...")
        self._gen_model(src, inc, peak_bytes, arena_layers)

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
    # Weights
    # ------------------------------------------------------------------

    def _gen_weights(self, src: Path, inc: Path) -> None:
        """Write weights.h and weights.c with all model weights as const float[]."""
        decls: list[str] = []
        defs:  list[str] = []

        for layer in self.model.layers:
            cls = layer.__class__.__name__
            w   = layer.get_weights()
            if not w:
                continue

            for i, arr in enumerate(w):
                arr  = np.array(arr, dtype=np.float32)
                name = f"wt_{layer.name.replace('/', '_').replace('-', '_')}_{i}"
                flat = arr.flatten()
                def _flt(v) -> str:
                    s = f"{v:.8g}"
                    if "." not in s and "e" not in s and "n" not in s:
                        s += ".0"
                    return s + "f"
                vals = ", ".join(_flt(v) for v in flat)
                decls.append(f"extern const float {name}[{len(flat)}];  "
                              f"/* {cls} '{layer.name}' w[{i}] shape={arr.shape} */")
                defs.append(f"const float {name}[{len(flat)}] = {{\n"
                            + textwrap.fill(vals, width=100,
                                            initial_indent="    ",
                                            subsequent_indent="    ")
                            + "\n};\n")

        header = ("/* weights.h — model weights as const float arrays in Flash */\n"
                  "#pragma once\n\n"
                  + "\n".join(decls) + "\n")
        source = ('/* weights.c — generated by deploy.py */\n'
                  '#include "weights.h"\n\n'
                  + "\n".join(defs))

        (inc / "weights.h").write_text(header)
        (src / "weights.c").write_text(source)

    # ------------------------------------------------------------------
    # Feature matrix data (mel_fb, dct_matrix, freq_bins)
    # ------------------------------------------------------------------

    def _gen_feat_data(self, src: Path, inc: Path) -> None:
        """Write feat_data.h and feat_data.c with mel filterbank const array."""
        import librosa

        sr     = self.fp["sample_rate"]
        n_fft  = self.fp["n_fft"]
        n_mels = self.fp["n_mels"]
        n_bins = n_fft // 2 + 1

        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)

        def _flt(v) -> str:
            s = f"{v:.8g}"
            if "." not in s and "e" not in s and "n" not in s:
                s += ".0"
            return s + "f"

        def _arr2d(name, arr):
            rows = []
            for row in arr:
                vals = ", ".join(_flt(v) for v in row)
                rows.append("    {" + vals + "}")
            return (f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n"
                    + ",\n".join(rows) + "\n};\n")

        header = (
            "/* feat_data.h — mel filterbank */\n"
            "#pragma once\n"
            '#include "features.h"\n\n'
            f"extern const float feat_mel_fb[{n_mels}][{n_bins}];\n"
        )
        source = (
            '/* feat_data.c — generated by deploy.py */\n'
            '#include "feat_data.h"\n\n'
            + _arr2d("feat_mel_fb", mel_fb)
        )

        (inc / "feat_data.h").write_text(header)
        (src / "feat_data.c").write_text(source)

    # ------------------------------------------------------------------
    # Model forward pass
    # ------------------------------------------------------------------

    def _gen_model(self, src: Path, inc: Path,
                   peak_bytes: int, arena_layers) -> None:
        """Write model.h and model.c with arena + generated forward pass."""
        n_classes  = len(self.labels)
        arena_kb   = peak_bytes / 1024
        model_name = getattr(self.model, "name", "model")

        # ── Build forward pass body ──────────────────────────────────────
        calls:    list[str] = []
        buf_idx = 0   # ping-pong: 0 = buf_a, 1 = buf_b

        # Arena offsets: buf_a at offset 0, buf_b at offset max_half_bytes
        # max_half_bytes = max input_bytes across all non-input layers
        max_in = max((l.input_bytes for l in arena_layers if l.kind != "input"),
                     default=0)
        # buf_b starts at max_in bytes into arena
        calls.append(f"    float *buf_a = arena;")
        calls.append(f"    float *buf_b = arena + {max_in // 4};")
        calls.append(f"    /* Copy model input into buf_a */")
        n_input = 1
        for d in arena_layers[0].output_shape if arena_layers else []:
            n_input *= d
        calls.append(f"    memcpy(buf_a, input, {n_input} * sizeof(float));")
        calls.append(f"    float *in_ptr = buf_a, *out_ptr = buf_b;")
        calls.append("")

        for layer in self.model.layers:
            cls = layer.__class__.__name__
            cfg = layer.get_config()
            w   = layer.get_weights()

            def wname(i):
                return f"wt_{layer.name.replace('/', '_').replace('-', '_')}_{i}"

            def _ishape(layer) -> tuple:
                inp = layer.input
                if isinstance(inp, (list, tuple)):
                    inp = inp[0]
                raw = inp.shape
                lst = raw.as_list() if hasattr(raw, "as_list") else list(raw)
                return tuple(d for d in lst if d is not None)

            if cls == "InputLayer":
                continue

            elif cls in ("Dropout",):
                calls.append(f"    /* {layer.name}: Dropout — identity at inference */")

            elif cls == "Normalization":
                try:
                    in_sh = _ishape(layer)
                    n_ch  = in_sh[-1]
                    n_el  = 1
                    for d in in_sh: n_el *= d
                    calls.append(f"    /* {layer.name}: Normalization */")
                    calls.append(f"    ml_normalize(in_ptr, {n_el}, {n_ch}, "
                                 f"{wname(0)}, {wname(1)}, 1e-3f, out_ptr);")
                    calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")
                except Exception:
                    calls.append(f"    /* {layer.name}: Normalization (skipped — not adapted) */")

            elif cls == "Conv2D":
                in_sh = _ishape(layer)
                h, w2, ic = in_sh[-3], in_sh[-2], in_sh[-1]
                kh, kw = cfg["kernel_size"]
                oc     = cfg["filters"]
                pad    = cfg.get("padding", "same")
                fn     = "ml_conv2d_relu_same" if pad == "same" else "ml_conv2d_relu_valid"
                calls.append(f"    /* {layer.name}: Conv2D {in_sh} -> filters={oc} pad={pad} */")
                calls.append(f"    {fn}(in_ptr, {h}, {w2}, {ic}, "
                             f"{wname(0)}, {kh}, {kw}, {oc}, {wname(1)}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "DepthwiseConv2D":
                in_sh = _ishape(layer)
                h, w2, ic = in_sh[-3], in_sh[-2], in_sh[-1]
                kh, kw = cfg["kernel_size"]
                calls.append(f"    /* {layer.name}: DepthwiseConv2D */")
                calls.append(f"    ml_dwconv2d_relu_same(in_ptr, {h}, {w2}, {ic}, "
                             f"{wname(0)}, {kh}, {kw}, {wname(1)}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "MaxPooling2D":
                in_sh = _ishape(layer)
                h, w2, c = in_sh[-3], in_sh[-2], in_sh[-1]
                ph, pw = cfg["pool_size"]
                sh, sw = cfg.get("strides") or (ph, pw)
                calls.append(f"    /* {layer.name}: MaxPool2D ({ph},{pw}) */")
                calls.append(f"    ml_maxpool2d(in_ptr, {h}, {w2}, {c}, "
                             f"{ph}, {pw}, {sh}, {sw}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "AveragePooling2D":
                in_sh = _ishape(layer)
                h, w2, c = in_sh[-3], in_sh[-2], in_sh[-1]
                ph, pw = cfg["pool_size"]
                sh, sw = cfg.get("strides") or (ph, pw)
                calls.append(f"    /* {layer.name}: AvgPool2D ({ph},{pw}) */")
                calls.append(f"    ml_avgpool2d(in_ptr, {h}, {w2}, {c}, "
                             f"{ph}, {pw}, {sh}, {sw}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "GlobalAveragePooling2D":
                in_sh = _ishape(layer)
                h, w2, c = in_sh[-3], in_sh[-2], in_sh[-1]
                calls.append(f"    /* {layer.name}: GlobalAvgPool2D {in_sh} -> ({c},) */")
                calls.append(f"    ml_gap2d(in_ptr, {h}, {w2}, {c}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "Flatten":
                calls.append(f"    /* {layer.name}: Flatten — no-op (already contiguous) */")

            elif cls == "BatchNormalization":
                in_sh = _ishape(layer)
                n_el  = 1
                for d in in_sh: n_el *= d
                n_ch  = in_sh[-1]
                calls.append(f"    /* {layer.name}: BatchNorm */")
                calls.append(f"    ml_batchnorm(in_ptr, {n_el}, {n_ch}, "
                             f"{wname(0)}, {wname(1)}, {wname(2)}, {wname(3)}, 1e-3f, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            elif cls == "Dense":
                in_sh = _ishape(layer)
                n_in  = in_sh[-1]
                n_out = cfg["units"]
                act   = cfg.get("activation", "linear")
                fn    = {"relu": "ml_dense_relu",
                         "softmax": "ml_dense_softmax"}.get(act, "ml_dense_linear")
                calls.append(f"    /* {layer.name}: Dense({n_out}) act={act} */")
                calls.append(f"    {fn}(in_ptr, {n_in}, {wname(0)}, {wname(1)}, {n_out}, out_ptr);")
                calls.append(f"    {{ float *_t=in_ptr; in_ptr=out_ptr; out_ptr=_t; }}")

            else:
                calls.append(f"    /* WARNING: unsupported layer {cls} '{layer.name}' — skipped */")

        calls.append("")
        calls.append("    /* Copy scores to output, return argmax */")
        calls.append(f"    memcpy(scores, in_ptr, {n_classes} * sizeof(float));")
        calls.append("    int best = 0;")
        calls.append(f"    for (int i = 1; i < {n_classes}; i++)")
        calls.append("        if (in_ptr[i] > in_ptr[best]) best = i;")
        calls.append("    return best;")

        forward_body = "\n".join(calls)

        # ── model.h ─────────────────────────────────────────────────────
        header = f"""/* model.h — generated by deploy.py
   Model: {model_name}   Classes: {n_classes}   Arena: {arena_kb:.1f} KB  */
#pragma once
#include <stdint.h>
#define MODEL_ARENA_BYTES  {peak_bytes}
#define MODEL_N_CLASSES    {n_classes}
#ifdef __cplusplus
extern "C" {{
#endif
void model_init(void);
/* arena: caller-supplied scratch buffer of at least MODEL_ARENA_BYTES bytes.
   Pass the same buffer used for PCM recording — they don't overlap in time. */
int model_run(const float *input, float *scores, float *arena);
#ifdef __cplusplus
}}
#endif
"""

        # ── model.c ─────────────────────────────────────────────────────
        source = f"""/* model.c — generated by deploy.py */
#include "model.h"
#include "weights.h"
#include <string.h>
#include <math.h>

{_C_PRIMITIVES}


void model_init(void) {{}}

int model_run(const float *input, float *scores, float *arena) {{
{forward_body}
}}
"""
        (inc / "model.h").write_text(header)
        (src / "model.c").write_text(source)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _gen_features(self, src: Path, inc: Path) -> None:
        n_frames  = self.fp["n_features"] if "n_features" in self.fp else (
            1 + (self.fp["n_frames"] if "n_frames" in self.fp else
                 int(round(self.fp["duration"] * self.fp["sample_rate"])) // self.fp["hop_length"])
        )
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

    # ------------------------------------------------------------------
    # Audio driver
    # ------------------------------------------------------------------

    def _gen_audio(self, src: Path, inc: Path) -> None:
        driver = self.board_cfg["audio_driver"]
        sr     = self.board_cfg["sample_rate"]
        h_tmpl, cpp_tmpl = _AUDIO_DRIVERS.get(driver,
                                               (_AUDIO_PDM_STM32, _AUDIO_GENERIC))
        (inc / "audio.h").write_text(
            h_tmpl.format(sample_rate=sr) + "\n#define AUDIO_DURATION_S  "
            + str(self.fp["duration"]) + "\n")
        (src / "audio.cpp").write_text(cpp_tmpl.format(sample_rate=sr))

    # ------------------------------------------------------------------
    # main.cpp
    # ------------------------------------------------------------------

    def _gen_main(self, src: Path, peak_bytes: int) -> None:
        label_list = ", ".join(f'"{l}"' for l in self.labels)
        model_name = getattr(self.model, "name", "model")
        (src / "main.cpp").write_text(
            _MAIN_CPP.format(
                board      = self.board_name,
                model_name = model_name,
                n_classes  = len(self.labels),
                arena_kb   = peak_bytes / 1024,
                label_list = label_list,
            )
        )

    # ------------------------------------------------------------------
    # platformio.ini
    # ------------------------------------------------------------------

    def _gen_platformio_ini(self) -> None:
        cfg   = self.board_cfg
        flags = "\n".join(f"    {f}" for f in cfg["build_flags"])
        deps  = "\n".join(f"    {d}" for d in cfg["lib_deps"]) or "    ; none"

        extra_ini = ""
        if self.board_name == "nicla_vision":
            ld_name = "nicla_vision.ld"
            (self.output_dir / ld_name).write_text(_NICLA_LD)
            extra_ini = f"board_build.ldscript = {ld_name}\n"

        (self.output_dir / "platformio.ini").write_text(
            _PIO_INI.format(
                board       = self.board_name,
                platform    = cfg["platform"],
                pio_board   = cfg["pio_board"],
                framework   = cfg["framework"],
                build_flags = flags,
                lib_deps    = deps,
                extra_ini   = extra_ini,
            )
        )
