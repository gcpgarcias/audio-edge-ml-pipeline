"""
Deep audio feature extractors.

Each extractor produces a 2-D or 1-D array per sample without time-axis
aggregation, preserving the temporal structure required by deep learning
models.

Extractor          name               Output shape         Suited for
-----------------  -----------------  -------------------  ---------------------
MelSpectrogram     audio_mel_spec     (n_mels, T)          CNN, Transformer
Waveform           audio_waveform     (n_samples,)         1-D CNN, RNN/LSTM
CQT                audio_cqt          (n_bins, T)          CNN (music/birdsong)
MFCCSequence       audio_mfcc_seq     (n_mfcc, T)          RNN/LSTM, Transformer

All extractors accept optional ``start_time`` / ``end_time`` keyword
arguments (forwarded from dataset loaders) for segment-level extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from ..base import BaseFeatureExtractor
from ..registry import register

_MIN_SAMPLES: int = 1  # absolute floor; librosa handles too-short audio


def _load_segment(
    path:        Path,
    sample_rate: int,
    start_time:  Optional[float],
    end_time:    Optional[float],
    min_duration: float = 0.1,
) -> np.ndarray:
    """Load (and optionally slice) a mono audio segment."""
    offset   = float(start_time) if start_time is not None else 0.0
    duration: Optional[float] = None
    if end_time is not None:
        duration = max(float(end_time) - offset, min_duration)

    audio, _ = librosa.load(
        path,
        sr=sample_rate,
        offset=offset,
        duration=duration,
        mono=True,
    )
    return audio


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


# ======================================================================
# Mel-spectrogram  (CNN / Transformer input)
# ======================================================================


@register
class AudioMelSpectrogram(BaseFeatureExtractor):
    """Log-scaled mel-spectrogram, normalized to [0, 1].

    Shape: ``(n_mels, time_frames)``

    This is the direct successor of the original ``AudioPreprocessor`` and
    can be used as a drop-in replacement for CNN-based training.

    Parameters
    ----------
    sample_rate: Target sample rate (Hz).
    n_mels:      Number of mel filter-bank bins.
    n_fft:       FFT window size in samples.
    hop_length:  Hop size in samples.
    duration:    Fixed output duration (seconds).  Audio is padded/trimmed.
                 *None* → keep natural segment length.
    """

    name         = "audio_mel_spec"
    feature_type = "deep"
    modality     = "audio"

    def __init__(
        self,
        sample_rate: int            = 16000,
        n_mels:      int            = 40,
        n_fft:       int            = 512,
        hop_length:  int            = 160,
        duration:    Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.duration    = duration

    def extract(
        self,
        sample_path: Path,
        start_time:  Optional[float] = None,
        end_time:    Optional[float] = None,
        **_kwargs,
    ) -> np.ndarray:
        audio = _load_segment(sample_path, self.sample_rate, start_time, end_time)

        if self.duration is not None:
            target = int(self.duration * self.sample_rate)
            audio  = _pad_or_trim(audio, target)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return _normalize(log_mel).astype(np.float32)


# ======================================================================
# Raw waveform  (1-D CNN / RNN / LSTM input)
# ======================================================================


@register
class AudioWaveform(BaseFeatureExtractor):
    """Raw PCM waveform, normalized to [-1, 1].

    Shape: ``(n_samples,)``

    A fixed ``duration`` is strongly recommended so that all samples in a
    batch share the same length.  Shorter segments are zero-padded.

    Parameters
    ----------
    sample_rate: Target sample rate (Hz).
    duration:    Fixed output duration (seconds).  *None* → natural length
                 (variable-length batching required downstream).
    """

    name         = "audio_waveform"
    feature_type = "deep"
    modality     = "audio"

    def __init__(
        self,
        sample_rate: int            = 16000,
        duration:    Optional[float] = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.duration    = duration

    def extract(
        self,
        sample_path: Path,
        start_time:  Optional[float] = None,
        end_time:    Optional[float] = None,
        **_kwargs,
    ) -> np.ndarray:
        audio = _load_segment(sample_path, self.sample_rate, start_time, end_time)

        if self.duration is not None:
            target = int(self.duration * self.sample_rate)
            audio  = _pad_or_trim(audio, target)

        # Normalize to [-1, 1] (standard for waveform models)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32)


# ======================================================================
# Constant-Q Transform  (CNN – especially effective for pitched sounds)
# ======================================================================


@register
class AudioCQT(BaseFeatureExtractor):
    """Magnitude Constant-Q Transform, log-scaled and normalized to [0, 1].

    Shape: ``(n_bins, time_frames)``

    The CQT uses geometrically spaced frequency bins, making it
    pitch-shift-invariant and well-suited to birdsong and music analysis.

    Parameters
    ----------
    sample_rate:  Target sample rate (Hz).
    hop_length:   Hop size in samples.
    n_bins:       Total number of CQT bins.
    bins_per_octave: Frequency resolution per octave (default 12 → semitone).
    fmin:         Lowest frequency (Hz).  *None* → librosa default (C1 ≈ 32 Hz).
    duration:     Fixed output duration (seconds).
    """

    name         = "audio_cqt"
    feature_type = "deep"
    modality     = "audio"

    def __init__(
        self,
        sample_rate:      int            = 22050,
        hop_length:       int            = 512,
        n_bins:           int            = 84,
        bins_per_octave:  int            = 12,
        fmin:             Optional[float] = None,
        duration:         Optional[float] = None,
    ) -> None:
        self.sample_rate     = sample_rate
        self.hop_length      = hop_length
        self.n_bins          = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin            = fmin
        self.duration        = duration

    def extract(
        self,
        sample_path: Path,
        start_time:  Optional[float] = None,
        end_time:    Optional[float] = None,
        **_kwargs,
    ) -> np.ndarray:
        audio = _load_segment(sample_path, self.sample_rate, start_time, end_time)

        if self.duration is not None:
            target = int(self.duration * self.sample_rate)
            audio  = _pad_or_trim(audio, target)

        cqt = np.abs(
            librosa.cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin,
            )
        )
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        return _normalize(log_cqt).astype(np.float32)


# ======================================================================
# MFCC sequence  (RNN / LSTM / Transformer sequence input)
# ======================================================================


@register
class AudioMFCCSequence(BaseFeatureExtractor):
    """MFCC sequence (without time-axis aggregation) for sequence models.

    Shape: ``(n_mfcc, time_frames)``

    Unlike :class:`AudioClassicalExtractor`, the time axis is preserved so
    that RNN/LSTM and Transformer encoders can model temporal dynamics.

    Parameters
    ----------
    sample_rate: Target sample rate (Hz).
    n_mfcc:      Number of MFCC coefficients.
    n_fft:       FFT window size.
    hop_length:  Hop size.
    duration:    Fixed output duration.  *None* → natural segment length.
    """

    name         = "audio_mfcc_seq"
    feature_type = "deep"
    modality     = "audio"

    def __init__(
        self,
        sample_rate: int            = 22050,
        n_mfcc:      int            = 40,
        n_fft:       int            = 1024,
        hop_length:  int            = 512,
        duration:    Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.duration    = duration

    def extract(
        self,
        sample_path: Path,
        start_time:  Optional[float] = None,
        end_time:    Optional[float] = None,
        **_kwargs,
    ) -> np.ndarray:
        audio = _load_segment(sample_path, self.sample_rate, start_time, end_time)

        if self.duration is not None:
            target = int(self.duration * self.sample_rate)
            audio  = _pad_or_trim(audio, target)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # Standardize per-coefficient (zero mean, unit variance)
        mean = mfcc.mean(axis=1, keepdims=True)
        std  = mfcc.std(axis=1, keepdims=True) + 1e-8
        return ((mfcc - mean) / std).astype(np.float32)