"""
Classical audio feature extractor.

Produces a fixed-length flat feature vector from a raw audio file (or a
time-stamped segment within it).  The vector is suitable for traditional ML
classifiers and clustering algorithms:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Feature groups (each aggregated as mean + standard deviation over time)
-----------------------------------------------------------------------
Group               Dimension   Description
------------------- ----------- ----------------------------------------
MFCCs               2 × n_mfcc  Mel-frequency cepstral coefficients
Delta MFCCs         2 × n_mfcc  First-order MFCC derivatives
Delta-delta MFCCs   2 × n_mfcc  Second-order MFCC derivatives
Spectral centroid   2           Weighted mean frequency
Spectral rolloff    2           Frequency below which 85 % of energy lies
Spectral bandwidth  2           Spread of energy around the centroid
Spectral contrast   2 × 7       Peak-valley contrast per sub-band
Spectral flatness   2           Tonality vs. noise measure
Chroma STFT         2 × 12      Energy per pitch class (C … B)
Zero-crossing rate  2           Rate of sign changes in the waveform
RMS energy          2           Root-mean-square amplitude
Tonnetz             2 × 6       Tonal centroid (fifths, minor/major thirds)
------------------- ----------- ----------------------------------------

Default total with n_mfcc=40: 3×(2×40) + 2+2+2+(2×7)+2+(2×12)+2+2+(2×6)
                             = 240 + 2+2+2+14+2+24+2+2+12 = 302 features
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
    """

    name         = "audio_classical"
    feature_type = "classical"
    modality     = "audio"

    def __init__(
        self,
        sample_rate: int   = 22050,
        n_mfcc:      int   = 40,
        n_fft:       int   = 1024,
        hop_length:  int   = 512,
        min_duration: float = _MIN_DURATION,
    ) -> None:
        self.sample_rate  = sample_rate
        self.n_mfcc       = n_mfcc
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.min_duration = min_duration

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

        Parameters
        ----------
        sample_path:
            Path to a WAV (or any librosa-readable) audio file.
        start_time:
            Segment onset in seconds (None → beginning of file).
        end_time:
            Segment offset in seconds (None → end of file).

        Returns
        -------
        np.ndarray
            1-D float32 feature vector of length
            ``3 * 2 * n_mfcc + 2 + 2 + 2 + 14 + 2 + 24 + 2 + 2 + 12``.
        """
        audio = self._load_segment(sample_path, start_time, end_time)
        return self._compute_features(audio).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Total number of features per sample."""
        return 3 * 2 * self.n_mfcc + 2 + 2 + 2 + 14 + 2 + 24 + 2 + 2 + 12

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

        # Guarantee at least min_duration worth of samples
        min_samples = int(self.min_duration * self.sample_rate)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        return audio

    def _compute_features(self, audio: np.ndarray) -> np.ndarray:
        sr  = self.sample_rate
        hop = self.hop_length
        n   = self.n_fft

        # ---- MFCCs and temporal derivatives ----
        mfcc    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc,
                                        n_fft=n, hop_length=hop)
        d_mfcc  = librosa.feature.delta(mfcc)
        dd_mfcc = librosa.feature.delta(mfcc, order=2)

        # ---- Spectral features ----
        spec_centroid  = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=n, hop_length=hop)
        spec_rolloff   = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n, hop_length=hop)
        spec_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=n, hop_length=hop)
        spec_contrast  = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=n, hop_length=hop)   # shape (7, T)
        spec_flatness  = librosa.feature.spectral_flatness(
            y=audio, n_fft=n, hop_length=hop)

        # ---- Chroma ----
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=n, hop_length=hop)   # shape (12, T)

        # ---- Zero-crossing rate ----
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop)

        # ---- RMS energy ----
        rms = librosa.feature.rms(y=audio, frame_length=n, hop_length=hop)

        # ---- Tonnetz ----
        # Requires harmonic component; librosa handles mono gracefully.
        harmonic = librosa.effects.harmonic(audio)
        tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)  # shape (6, T)

        # ---- Aggregate: mean + std over time axis ----
        parts = [
            _mean_std(mfcc),             # 2 * n_mfcc
            _mean_std(d_mfcc),           # 2 * n_mfcc
            _mean_std(dd_mfcc),          # 2 * n_mfcc
            _scalar_mean_std(spec_centroid),   # 2
            _scalar_mean_std(spec_rolloff),    # 2
            _scalar_mean_std(spec_bandwidth),  # 2
            _mean_std(spec_contrast),    # 2 * 7 = 14
            _scalar_mean_std(spec_flatness),   # 2
            _mean_std(chroma),           # 2 * 12 = 24
            _scalar_mean_std(zcr),       # 2
            _scalar_mean_std(rms),       # 2
            _mean_std(tonnetz),          # 2 * 6 = 12
        ]

        return np.concatenate(parts)