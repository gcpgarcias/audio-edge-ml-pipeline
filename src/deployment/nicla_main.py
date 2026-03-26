"""
Nicla Vision inference script — PCA+SVM audio event classification.

No TFLite required.  Inference is pure ulab matrix math:
    record → MFCC + spectral features → StandardScaler → PCA → linear SVM (OvO)

Deploy bundle (copy ALL files to the board root via OpenMV IDE):
    scaler_mean.npy
    scaler_scale.npy
    pca_mean.npy
    pca_components.npy
    svm_coef.npy
    svm_intercept.npy
    mel_fb.npy
    dct_matrix.npy
    freq_bins.npy
    label_names.json
    feature_params.json

Generate the bundle with:
    python -m src.deployment.export_svm \\
        --model-dir  data/models/tuned/fsc22_12_classes_nicla_pca_svm_sweep \\
        --features-dir data/processed/fsc22_classical_nicla_train \\
        --output deploy/nicla_svm

LED colour coding
-----------------
    Blue   → ready / waiting
    Orange → recording
    Yellow → extracting features
    Green  → result ready
    Red    → error

Hardware: Arduino Nicla Vision (STM32H747, MP34DT05 PDM mic)
Firmware: OpenMV >= 4.5
"""

import gc
import json
import time

from ulab import numpy as np
import audio
import pyb

npfft = np.fft


# ---------------------------------------------------------------------------
# LED helpers
# ---------------------------------------------------------------------------

_led_r = pyb.LED(1)
_led_g = pyb.LED(2)
_led_b = pyb.LED(3)


def _led(r=False, g=False, b=False):
    _led_r.on() if r else _led_r.off()
    _led_g.on() if g else _led_g.off()
    _led_b.on() if b else _led_b.off()


def _led_orange(): _led(r=True,  g=True,  b=False)
def _led_yellow(): _led(r=True,  g=False, b=False)   # Nicla has no true yellow; use red
def _led_green():  _led(r=False, g=True,  b=False)
def _led_blue():   _led(r=False, g=False, b=True)
def _led_red():    _led(r=True,  g=False, b=False)
def _led_off():    _led()


# ---------------------------------------------------------------------------
# Boot: load deployment artifacts
# ---------------------------------------------------------------------------

_led_orange()
print("Loading deployment artifacts …")

try:
    with open("feature_params.json") as f:
        fp = json.load(f)
except OSError:
    print("ERROR: feature_params.json not found — copy deploy bundle to board root.")
    _led_red()
    raise

SAMPLE_RATE = int(fp["sample_rate"])
N_MFCC      = int(fp["n_mfcc"])
N_FFT       = int(fp["n_fft"])
HOP_LENGTH  = int(fp["hop_length"])
N_MELS      = int(fp["n_mels"])
DURATION    = float(fp["duration"])
N_SAMPLES   = int(DURATION * SAMPLE_RATE)
N_BINS      = N_FFT // 2 + 1
N_PCA       = int(fp["n_pca"])
N_CLASSES   = int(fp["n_classes"])

print(f"sr={SAMPLE_RATE}  n_fft={N_FFT}  hop={HOP_LENGTH}  "
      f"n_mels={N_MELS}  n_mfcc={N_MFCC}  duration={DURATION}s")

with open("label_names.json") as f:
    labels = json.load(f)
print(f"Labels ({len(labels)}): {labels}")

# Load model parameters in order of increasing size to maximise contiguous heap
scaler_mean   = np.load("scaler_mean.npy")      # (n_features,)
scaler_scale  = np.load("scaler_scale.npy")     # (n_features,)
pca_mean      = np.load("pca_mean.npy")         # (n_features,)
svm_intercept = np.load("svm_intercept.npy")    # (n_pairs,)
freq_bins     = np.load("freq_bins.npy")        # (N_BINS,)
dct_matrix    = np.load("dct_matrix.npy")       # (N_MFCC, N_MELS)
svm_coef      = np.load("svm_coef.npy")         # (n_pairs, N_PCA)
pca_comps     = np.load("pca_components.npy")   # (N_PCA, n_features)
gc.collect()
mel_fb        = np.load("mel_fb.npy")           # (N_MELS, N_BINS)  — largest array
gc.collect()

print(f"Free heap after artifact load: {gc.mem_free()} bytes")

# Pre-compute Hann window (N_FFT samples)
_hann = np.array(
    [0.5 * (1.0 - np.cos(2.0 * 3.14159265 * i / N_FFT)) for i in range(N_FFT)],
    dtype=np.float32,
)

_led_blue()
print("Ready.")


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_audio() -> "np.ndarray":
    """Record DURATION seconds of mono PCM at SAMPLE_RATE.

    Returns float32 array in [-1, 1] of length N_SAMPLES.
    """
    _buf  = bytearray(N_SAMPLES * 2)   # int16 PCM
    _pos  = [0]
    _done = [False]

    def _cb(pcm):
        if _done[0]:
            return
        n = min(len(pcm), N_SAMPLES * 2 - _pos[0])
        _buf[_pos[0]:_pos[0] + n] = pcm[:n]
        _pos[0] += n
        if _pos[0] >= N_SAMPLES * 2:
            _done[0] = True

    audio.init(channels=1, frequency=SAMPLE_RATE, gain_db=24, highpass=0.9883)
    audio.start_streaming(_cb)
    while not _done[0]:
        pass
    audio.stop_streaming()

    pcm_i16 = np.frombuffer(_buf, dtype=np.int16)
    return pcm_i16.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Feature extraction  (92 dims matching AudioClassicalExtractor)
# ---------------------------------------------------------------------------
#
# Feature vector layout (mean then std over frames, canonical extractor order):
#
#   mfcc              40×2 = 80
#   spectral_centroid  1×2 =  2
#   spectral_rolloff   1×2 =  2
#   spectral_bandwidth 1×2 =  2
#   spectral_flatness  1×2 =  2
#   zcr                1×2 =  2
#   rms                1×2 =  2
#   ─────────────────────────────
#   TOTAL                   92
#
# All spectral features are derived from the same per-frame power spectrum,
# so the STFT is computed only once per frame.
# ---------------------------------------------------------------------------

def extract_features(pcm: "np.ndarray") -> "np.ndarray":
    """Compute 92-dim classical feature vector from raw audio.

    Uses online mean/variance to avoid storing full (N_MFCC, n_frames) matrix.
    """
    n_frames = (len(pcm) - N_FFT) // HOP_LENGTH + 1

    # Online accumulators: sum and sum-of-squares for each scalar series
    mfcc_sum    = np.zeros(N_MFCC, dtype=np.float32)
    mfcc_sum_sq = np.zeros(N_MFCC, dtype=np.float32)

    c_sum = c_sum_sq = 0.0   # centroid
    ro_sum = ro_sum_sq = 0.0  # rolloff
    bw_sum = bw_sum_sq = 0.0  # bandwidth
    fl_sum = fl_sum_sq = 0.0  # flatness
    z_sum  = z_sum_sq  = 0.0  # zcr
    r_sum  = r_sum_sq  = 0.0  # rms

    for i in range(n_frames):
        start    = i * HOP_LENGTH
        frame    = pcm[start:start + N_FFT]
        windowed = frame * _hann

        # ── Power spectrum ──────────────────────────────────────────────────
        spectrum = npfft.fft(windowed)
        mag      = np.abs(spectrum[:N_BINS])     # one-sided magnitude
        power    = mag ** 2                       # one-sided power

        # ── MFCCs ──────────────────────────────────────────────────────────
        mel_e   = np.dot(mel_fb, power)           # (N_MELS,)
        log_mel = np.log(mel_e + 1e-10)           # (N_MELS,)
        mfcc_v  = np.dot(dct_matrix, log_mel)     # (N_MFCC,)
        mfcc_sum    += mfcc_v
        mfcc_sum_sq += mfcc_v ** 2

        # ── Spectral centroid ───────────────────────────────────────────────
        mag_sum = float(np.sum(mag)) + 1e-10
        c = float(np.sum(freq_bins * mag)) / mag_sum
        c_sum    += c
        c_sum_sq += c * c

        # ── Spectral rolloff (85% energy threshold) ─────────────────────────
        threshold = 0.85 * float(np.sum(power))
        cum = 0.0
        rb  = N_BINS - 1
        for b in range(N_BINS):
            cum += float(power[b])
            if cum >= threshold:
                rb = b
                break
        ro = float(freq_bins[rb])
        ro_sum    += ro
        ro_sum_sq += ro * ro

        # ── Spectral bandwidth ──────────────────────────────────────────────
        bw = float(np.sqrt(np.sum(((freq_bins - c) ** 2) * mag) / mag_sum + 1e-10))
        bw_sum    += bw
        bw_sum_sq += bw * bw

        # ── Spectral flatness ───────────────────────────────────────────────
        log_mean   = float(np.mean(np.log(power + 1e-10)))
        arith_mean = float(np.mean(power)) + 1e-10
        # exp via Taylor is inaccurate; import math is unavailable in OpenMV
        # Use the identity: e^x = 2^(x/ln2) approximated via ** operator
        fl = (2.718281828 ** log_mean) / arith_mean
        fl_sum    += fl
        fl_sum_sq += fl * fl

        # ── Zero-crossing rate ──────────────────────────────────────────────
        z = float(np.sum(frame[:-1] * frame[1:] < 0)) / float(N_FFT)
        z_sum    += z
        z_sum_sq += z * z

        # ── RMS energy ─────────────────────────────────────────────────────
        r = float(np.sqrt(np.mean(frame ** 2)))
        r_sum    += r
        r_sum_sq += r * r

    # ── Aggregate: mean and std from online accumulators ─────────────────────
    n = float(n_frames)

    def _mean_std_mfcc():
        mean = mfcc_sum / n
        std  = np.sqrt(np.maximum(mfcc_sum_sq / n - mean ** 2,
                                   np.zeros(N_MFCC, dtype=np.float32)))
        return mean, std

    def _ms(s, s2):
        mean = s / n
        var  = s2 / n - mean * mean
        std  = (var ** 0.5) if var > 0 else 0.0
        return mean, std

    mfcc_mean, mfcc_std = _mean_std_mfcc()
    c_m,  c_s  = _ms(c_sum,  c_sum_sq)
    ro_m, ro_s = _ms(ro_sum, ro_sum_sq)
    bw_m, bw_s = _ms(bw_sum, bw_sum_sq)
    fl_m, fl_s = _ms(fl_sum, fl_sum_sq)
    z_m,  z_s  = _ms(z_sum,  z_sum_sq)
    r_m,  r_s  = _ms(r_sum,  r_sum_sq)

    feat = np.concatenate([
        mfcc_mean, mfcc_std,                                    # 80
        np.array([c_m,  c_s],  dtype=np.float32),              #  2
        np.array([ro_m, ro_s], dtype=np.float32),              #  2
        np.array([bw_m, bw_s], dtype=np.float32),              #  2
        np.array([fl_m, fl_s], dtype=np.float32),              #  2
        np.array([z_m,  z_s],  dtype=np.float32),              #  2
        np.array([r_m,  r_s],  dtype=np.float32),              #  2
    ])
    return feat   # (92,)


# ---------------------------------------------------------------------------
# PCA + linear SVM inference
# ---------------------------------------------------------------------------

def predict(feat: "np.ndarray") -> tuple:
    """Classify a 92-dim feature vector.

    Pipeline: StandardScaler → PCA → linear SVM OvO voting.

    Returns
    -------
    (label: str, confidence: float, class_idx: int)
    confidence is normalised vote fraction in [0, 1].
    """
    # StandardScaler
    x = (feat - scaler_mean) / scaler_scale        # (n_features,)

    # PCA projection
    x_pca = np.dot(pca_comps, x - pca_mean)        # (N_PCA,)

    # Linear SVM — OvO voting
    # Pairs iterate in sklearn order: (0,1),(0,2),…,(0,K),(1,2),…,(K-1,K)
    votes = [0] * N_CLASSES
    k = 0
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            d = float(np.dot(svm_coef[k], x_pca)) + float(svm_intercept[k])
            if d > 0:
                votes[i] += 1
            else:
                votes[j] += 1
            k += 1

    idx        = votes.index(max(votes))
    confidence = max(votes) / float(N_CLASSES - 1)   # normalise to [0, 1]
    return labels[idx], confidence, idx


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

while True:
    _led_blue()
    print("\n─── Listening … press Ctrl+C to stop ───")
    time.sleep_ms(500)

    # 1. Record
    _led_orange()
    print(f"Recording {DURATION:.0f}s …")
    t0  = time.ticks_ms()
    pcm = record_audio()
    print(f"  recorded in {time.ticks_diff(time.ticks_ms(), t0)} ms")

    # 2. Extract features
    _led_yellow()
    print("Extracting features …")
    t0   = time.ticks_ms()
    feat = extract_features(pcm)
    print(f"  done in {time.ticks_diff(time.ticks_ms(), t0)} ms  dim={len(feat)}")

    # 3. Classify
    t0               = time.ticks_ms()
    label, conf, idx = predict(feat)
    print(f"  classify in {time.ticks_diff(time.ticks_ms(), t0)} ms")

    # 4. Report
    _led_green()
    print(f"  → {label}  (confidence {conf * 100:.0f}%)")

    time.sleep_ms(1000)
