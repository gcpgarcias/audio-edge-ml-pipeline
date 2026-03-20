"""
Classical audio feature extractor.

Produces a fixed-length flat feature vector from a raw audio file (or a
time-stamped segment within it).  The vector is suitable for traditional ML
classifiers and clustering algorithms:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Feature groups (aggregated over time according to ``aggregations``)
-----------------------------------------------------------------------
Group               Key                 Raw dim   Description
------------------- ------------------- --------- ----------------------------------------
MFCCs               mfcc                n_mfcc    Mel-frequency cepstral coefficients
Delta MFCCs         delta_mfcc          n_mfcc    First-order MFCC derivatives
Delta-delta MFCCs   delta2_mfcc         n_mfcc    Second-order MFCC derivatives
Spectral centroid   spectral_centroid   1         Weighted mean frequency
Spectral rolloff    spectral_rolloff    1         Frequency below which 85 % of energy lies
Spectral bandwidth  spectral_bandwidth  1         Spread of energy around the centroid
Spectral contrast   spectral_contrast   7         Peak-valley contrast per sub-band
Spectral flatness   spectral_flatness   1         Tonality vs. noise measure
Chroma STFT         chroma              12        Energy per pitch class (C … B)
Zero-crossing rate  zcr                 1         Rate of sign changes in the waveform
RMS energy          rms                 1         Root-mean-square amplitude
Tonnetz             tonnetz             6         Tonal centroid (fifths, minor/major thirds)
------------------- ------------------- --------- ----------------------------------------

Each group contributes  raw_dim × len(aggregations)  values to the vector.

Default (aggregations=["mean","std"], n_mfcc=40, all groups):
    3×(2×40) + (2×1)×6 + (2×7) + (2×12) + (2×6) = 302 features

Mean-only (aggregations=["mean"], n_mfcc=40, all groups):
    3×40 + 6 + 7 + 12 + 6 = 151 features

Lean vector (no delta MFCCs, no Tonnetz, default aggregations):
    features: [mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth,
               spectral_contrast, spectral_flatness, chroma, zcr, rms]
    → 2×40 + 2+2+2+14+2+24+2+2 = 130 features

Lean + mean-only:
    → 40 + 1+1+1+7+1+12+1+1 = 65 features
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

# Raw dimension of each group *before* aggregation (independent of n_mfcc).
# MFCC-family groups are handled separately as they depend on n_mfcc.
_RAW_DIMS: dict[str, int] = {
    "spectral_centroid":  1,
    "spectral_rolloff":   1,
    "spectral_bandwidth": 1,
    "spectral_contrast":  7,
    "spectral_flatness":  1,
    "chroma":             12,
    "zcr":                1,
    "rms":                1,
    "tonnetz":            6,
}

# Valid temporal aggregation functions.
_ALL_AGGREGATIONS: list[str] = ["mean", "std"]


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

    aggregations:
        Temporal aggregation functions applied to each feature group.
        Defaults to ``["mean", "std"]``.  Valid values: ``"mean"``, ``"std"``.
        Use ``["mean"]`` to halve the feature vector (drops temporal-dynamics
        signal; useful for ablation or edge devices with tight memory budgets).
        Order is preserved: ``["mean", "std"]`` always places mean before std.
    """

    name         = "audio_classical"
    feature_type = "classical"
    modality     = "audio"

    def __init__(
        self,
        sample_rate:   int              = 22050,
        n_mfcc:        int              = 40,
        n_fft:         int              = 1024,
        hop_length:    int              = 512,
        min_duration:  float            = _MIN_DURATION,
        features:      Optional[list[str]] = None,
        aggregations:  Optional[list[str]] = None,
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

        if aggregations is None:
            self.aggregations = list(_ALL_AGGREGATIONS)
        else:
            unknown = set(aggregations) - set(_ALL_AGGREGATIONS)
            if unknown:
                raise ValueError(
                    f"Unknown aggregation(s): {sorted(unknown)}. "
                    f"Valid values: {_ALL_AGGREGATIONS}"
                )
            if not aggregations:
                raise ValueError("aggregations must contain at least one value.")
            # Preserve canonical ordering.
            self.aggregations = [a for a in _ALL_AGGREGATIONS if a in set(aggregations)]

        self._agg_set = set(self.aggregations)

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
        """Total number of features for the current configuration."""
        n_agg = len(self.aggregations)
        total = 0
        for key in self.features:
            if key in ("mfcc", "delta_mfcc", "delta2_mfcc"):
                total += n_agg * self.n_mfcc
            else:
                total += n_agg * _RAW_DIMS[key]
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _agg(self, x: np.ndarray, *, scalar: bool = False) -> np.ndarray:
        """Aggregate a frame-level array according to ``self.aggregations``.

        Parameters
        ----------
        x:
            2-D array of shape ``(n_features, n_frames)``.
        scalar:
            If ``True``, the array has a single feature row (or is 1-D) and
            the result is a flat vector of length ``len(aggregations)``.
            If ``False``, mean/std are computed along the time axis and
            concatenated, giving a vector of length
            ``n_features × len(aggregations)``.
        """
        parts = []
        if "mean" in self._agg_set:
            parts.append(
                np.array([float(x.mean())]) if scalar else x.mean(axis=1)
            )
        if "std" in self._agg_set:
            parts.append(
                np.array([float(x.std())]) if scalar else x.std(axis=1)
            )
        return np.concatenate(parts)

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
            "mfcc":               lambda: self._agg(mfcc),
            "delta_mfcc":         lambda: self._agg(d_mfcc),
            "delta2_mfcc":        lambda: self._agg(dd_mfcc),
            "spectral_centroid":  lambda: self._agg(spec_centroid,  scalar=True),
            "spectral_rolloff":   lambda: self._agg(spec_rolloff,   scalar=True),
            "spectral_bandwidth": lambda: self._agg(spec_bandwidth, scalar=True),
            "spectral_contrast":  lambda: self._agg(spec_contrast),
            "spectral_flatness":  lambda: self._agg(spec_flatness,  scalar=True),
            "chroma":             lambda: self._agg(chroma),
            "zcr":                lambda: self._agg(zcr,            scalar=True),
            "rms":                lambda: self._agg(rms,            scalar=True),
            "tonnetz":            lambda: self._agg(tonnetz),
        }

        parts = [_aggregators[key]() for key in self.features]
        return np.concatenate(parts)
