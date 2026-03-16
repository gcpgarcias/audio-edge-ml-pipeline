"""
Classical audio feature extractor.

Produces a fixed-length flat feature vector from a raw audio file (or a
time-stamped segment within it).  The vector is suitable for traditional ML
classifiers and clustering algorithms:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Feature groups (each aggregated as mean + standard deviation over time)
-----------------------------------------------------------------------
Group               Key                 Dimension   Description
------------------- ------------------- ----------- ----------------------------------------
MFCCs               mfcc                2 × n_mfcc  Mel-frequency cepstral coefficients
Delta MFCCs         delta_mfcc          2 × n_mfcc  First-order MFCC derivatives
Delta-delta MFCCs   delta2_mfcc         2 × n_mfcc  Second-order MFCC derivatives
Spectral centroid   spectral_centroid   2           Weighted mean frequency
Spectral rolloff    spectral_rolloff    2           Frequency below which 85 % of energy lies
Spectral bandwidth  spectral_bandwidth  2           Spread of energy around the centroid
Spectral contrast   spectral_contrast   2 × 7       Peak-valley contrast per sub-band
Spectral flatness   spectral_flatness   2           Tonality vs. noise measure
Chroma STFT         chroma              2 × 12      Energy per pitch class (C … B)
Zero-crossing rate  zcr                 2           Rate of sign changes in the waveform
RMS energy          rms                 2           Root-mean-square amplitude
Tonnetz             tonnetz             2 × 6       Tonal centroid (fifths, minor/major thirds)
------------------- ------------------- ----------- ----------------------------------------

Default total with n_mfcc=40 (all groups):
    3×(2×40) + 2+2+2+(2×7)+2+(2×12)+2+2+(2×6) = 302 features

Selecting a subset via the ``features`` parameter drops the unwanted groups
and recomputes the dimension accordingly.  Example lean vector (no delta
MFCCs, no Tonnetz):

    features: [mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth,
               spectral_contrast, spectral_flatness, chroma, zcr, rms]
    → 2×40 + 2+2+2+14+2+24+2+2 = 130 features
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from ..base import BaseFeatureExtractor
from ..registry import register

# Minimum audio segment duration (seconds) used when start_time == end_time
# or the requested segment is unreasonably short.
_MIN_DURATION: float = 0.1

# Ordered list of all valid feature group keys.
_ALL_FEATURES: list[str] = [
    "mfcc",
    "delta_mfcc",
    "delta2_mfcc",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_flatness",
    "chroma",
    "zcr",
    "rms",
    "tonnetz",
]

# Fixed dimension of each group (independent of n_mfcc where applicable).
_FIXED_DIMS: dict[str, int] = {
    "spectral_centroid": 2,
    "spectral_rolloff":  2,
    "spectral_bandwidth": 2,
    "spectral_contrast": 14,  # 2 × 7 sub-bands
    "spectral_flatness": 2,
    "chroma":            24,  # 2 × 12 pitch classes
    "zcr":               2,
    "rms":               2,
    "tonnetz":           12,  # 2 × 6
}


def _mean_std(x: np.ndarray) -> np.ndarray:
    """Concatenate column-wise mean and std of a 2-D array → 1-D vector."""
    return np.concatenate([x.mean(axis=1), x.std(axis=1)])


def _scalar_mean_std(x: np.ndarray) -> np.ndarray:
    """Concatenate mean and std of a 1- or 2-D array → [mean, std]."""
    return np.array([float(x.mean()), float(x.std())])


@register
class AudioClassicalExtractor(BaseFeatureExtractor):
    """Flat classical audio features suitable for sklearn estimators.

    Parameters
    ----------
    sample_rate:
        Target sample rate (Hz).  Audio is resampled if necessary.
    n_mfcc:
        Number of MFCC coefficients to compute.
    n_fft:
        FFT window size in samples.
    hop_length:
        Hop size in samples between successive frames.
    min_duration:
        Minimum segment duration (seconds).  Segments shorter than this are
        zero-padded.
    features:
        List of feature group keys to include.  Order is preserved.
        Defaults to all groups.  Valid keys::

            mfcc  delta_mfcc  delta2_mfcc
            spectral_centroid  spectral_rolloff  spectral_bandwidth
            spectral_contrast  spectral_flatness
            chroma  zcr  rms  tonnetz
    """

    name         = "audio_classical"
    feature_type = "classical"
    modality     = "audio"

    def __init__(
        self,
        sample_rate:  int   = 22050,
        n_mfcc:       int   = 40,
        n_fft:        int   = 1024,
        hop_length:   int   = 512,
        min_duration: float = _MIN_DURATION,
        features:     Optional[list[str]] = None,
    ) -> None:
        self.sample_rate  = sample_rate
        self.n_mfcc       = n_mfcc
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.min_duration = min_duration

        if features is None:
            self.features = list(_ALL_FEATURES)
        else:
            unknown = set(features) - set(_ALL_FEATURES)
            if unknown:
                raise ValueError(
                    f"Unknown feature group(s): {sorted(unknown)}. "
                    f"Valid keys: {_ALL_FEATURES}"
                )
            # Preserve canonical ordering regardless of input order.
            self.features = [k for k in _ALL_FEATURES if k in set(features)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        sample_path: Path,
        start_time:  Optional[float] = None,
        end_time:    Optional[float] = None,
        **_kwargs,
    ) -> np.ndarray:
        """Extract classical features from *sample_path*.

        Returns
        -------
        np.ndarray
            1-D float32 feature vector of length :attr:`feature_dim`.
        """
        audio = self._load_segment(sample_path, start_time, end_time)
        return self._compute_features(audio).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Total number of features for the current ``features`` selection."""
        mfcc_dim = 2 * self.n_mfcc
        total = 0
        for key in self.features:
            if key in ("mfcc", "delta_mfcc", "delta2_mfcc"):
                total += mfcc_dim
            else:
                total += _FIXED_DIMS[key]
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_segment(
        self,
        path:       Path,
        start_time: Optional[float],
        end_time:   Optional[float],
    ) -> np.ndarray:
        offset   = float(start_time) if start_time is not None else 0.0
        duration: Optional[float] = None
        if end_time is not None:
            duration = max(float(end_time) - offset, self.min_duration)

        audio, _ = librosa.load(
            path,
            sr=self.sample_rate,
            offset=offset,
            duration=duration,
            mono=True,
        )

        # Guarantee enough samples for:
        #   • n_fft — at least one STFT frame
        #   • delta width=9 — at least 9 MFCC frames (requires (9-1)*hop_length samples with center=True)
        min_samples = max(
            int(self.min_duration * self.sample_rate),
            self.n_fft,
            8 * self.hop_length,  # (delta_width - 1) * hop_length
        )
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        return audio

    def _compute_features(self, audio: np.ndarray) -> np.ndarray:
        sr  = self.sample_rate
        hop = self.hop_length
        n   = self.n_fft

        active = set(self.features)

        # ---- Compute only what is needed ----

        # MFCCs (base; also required for delta/delta2 and tonnetz via chroma)
        mfcc = None
        if active & {"mfcc", "delta_mfcc", "delta2_mfcc"}:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc,
                                        n_fft=n, hop_length=hop)

        d_mfcc = None
        if "delta_mfcc" in active:
            d_mfcc = librosa.feature.delta(mfcc)

        dd_mfcc = None
        if "delta2_mfcc" in active:
            dd_mfcc = librosa.feature.delta(mfcc, order=2)

        spec_centroid = None
        if "spectral_centroid" in active:
            spec_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=n, hop_length=hop)

        spec_rolloff = None
        if "spectral_rolloff" in active:
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, n_fft=n, hop_length=hop)

        spec_bandwidth = None
        if "spectral_bandwidth" in active:
            spec_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sr, n_fft=n, hop_length=hop)

        spec_contrast = None
        if "spectral_contrast" in active:
            spec_contrast = librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_fft=n, hop_length=hop)

        spec_flatness = None
        if "spectral_flatness" in active:
            spec_flatness = librosa.feature.spectral_flatness(
                y=audio, n_fft=n, hop_length=hop)

        # Chroma is needed for both chroma and tonnetz
        chroma = None
        if active & {"chroma", "tonnetz"}:
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=sr, n_fft=n, hop_length=hop)

        zcr = None
        if "zcr" in active:
            zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop)

        rms = None
        if "rms" in active:
            rms = librosa.feature.rms(y=audio, frame_length=n, hop_length=hop)

        tonnetz = None
        if "tonnetz" in active:
            tonnetz = librosa.feature.tonnetz(chroma=chroma)

        # ---- Aggregate in canonical order ----
        _aggregators = {
            "mfcc":               lambda: _mean_std(mfcc),
            "delta_mfcc":         lambda: _mean_std(d_mfcc),
            "delta2_mfcc":        lambda: _mean_std(dd_mfcc),
            "spectral_centroid":  lambda: _scalar_mean_std(spec_centroid),
            "spectral_rolloff":   lambda: _scalar_mean_std(spec_rolloff),
            "spectral_bandwidth": lambda: _scalar_mean_std(spec_bandwidth),
            "spectral_contrast":  lambda: _mean_std(spec_contrast),
            "spectral_flatness":  lambda: _scalar_mean_std(spec_flatness),
            "chroma":             lambda: _mean_std(chroma),
            "zcr":                lambda: _scalar_mean_std(zcr),
            "rms":                lambda: _scalar_mean_std(rms),
            "tonnetz":            lambda: _mean_std(tonnetz),
        }

        parts = [_aggregators[key]() for key in self.features]
        return np.concatenate(parts)
