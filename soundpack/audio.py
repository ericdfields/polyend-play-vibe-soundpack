"""Audio analysis using librosa."""

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def _get_safe_n_fft(signal_length: int, default: int = 2048) -> int:
    """Get an n_fft value that fits the signal length.

    Args:
        signal_length: Number of samples in the signal.
        default: Default n_fft value to use if signal is long enough.

    Returns:
        Power of 2 that fits the signal, or default if signal is long enough.
    """
    if signal_length >= default:
        return default

    # Find largest power of 2 that fits the signal
    n_fft = 1
    while n_fft * 2 <= signal_length:
        n_fft *= 2

    # Minimum useful n_fft is 64
    return max(64, n_fft)


def get_audio_info(file_path: str | Path) -> dict[str, Any]:
    """Get basic audio file information.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with duration_ms, sample_rate, channels, bit_depth.

    Raises:
        Exception: If file cannot be read as audio.
    """
    info = sf.info(file_path)

    # Map subtype to bit depth
    subtype_bits = {
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    bit_depth = subtype_bits.get(info.subtype, 16)

    return {
        "duration_ms": int(info.duration * 1000),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "bit_depth": bit_depth,
    }


def detect_bpm(file_path: str | Path) -> tuple[float | None, float]:
    """Detect BPM (tempo) of an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple of (bpm, confidence). BPM may be None if detection fails.
        Confidence is 0-1.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Get safe n_fft for this signal length
        n_fft = _get_safe_n_fft(len(y))
        hop_length = n_fft // 4

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )

        # Handle newer librosa versions that return an array
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) > 0 else None
        else:
            tempo = float(tempo)

        if tempo is None or tempo == 0:
            return None, 0.0

        # Calculate confidence based on onset strength consistency
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        if onset_env.mean() > 0:
            confidence = min(1.0, float(onset_env.std() / onset_env.mean()))
        else:
            confidence = 0.0

        return tempo, confidence

    except Exception:
        return None, 0.0


def detect_key(file_path: str | Path) -> tuple[str | None, float]:
    """Detect the musical key of an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple of (key_string, confidence). Key is like "C major" or "A minor".
        May return (None, 0) if detection fails.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Get safe hop length for this signal length
        n_fft = _get_safe_n_fft(len(y))
        hop_length = n_fft // 4

        # Use chroma_stft consistently as it accepts n_fft parameter
        # and avoids warnings from librosa's internal CQT processing
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )

        # Key names
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Krumhansl-Kessler key profiles
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # Normalize profiles
        major_profile = major_profile / major_profile.sum()
        minor_profile = minor_profile / minor_profile.sum()

        # Get average chroma distribution
        chroma_avg = chroma.mean(axis=1)
        chroma_avg = chroma_avg / (chroma_avg.sum() + 1e-8)

        # Correlate with all possible keys
        best_corr = -1
        best_key = None

        for i in range(12):
            # Rotate profiles to test each key
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)

            major_corr = np.corrcoef(chroma_avg, major_rot)[0, 1]
            minor_corr = np.corrcoef(chroma_avg, minor_rot)[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = f"{key_names[i]} major"

            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = f"{key_names[i]} minor"

        confidence = max(0, min(1, (best_corr + 1) / 2))  # Normalize to 0-1

        return best_key, float(confidence)

    except Exception:
        return None, 0.0


def extract_spectral_features(file_path: str | Path) -> dict[str, float]:
    """Extract spectral features from audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with spectral_centroid, onset_strength, rms_energy.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Get safe n_fft for this signal length
    n_fft = _get_safe_n_fft(len(y))
    hop_length = n_fft // 4

    # Spectral centroid (brightness indicator)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    centroid_mean = float(spectral_centroid.mean())

    # Onset strength (transient sharpness)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    onset_mean = float(onset_env.mean())

    # RMS energy (perceived loudness)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    rms_mean = float(rms.mean())

    return {
        "spectral_centroid": centroid_mean,
        "onset_strength": onset_mean,
        "rms_energy": rms_mean,
    }


def is_likely_loop(file_path: str | Path, min_duration_ms: int = 1000) -> bool:
    """Determine if a sample is likely a loop based on duration and content.

    Args:
        file_path: Path to the audio file.
        min_duration_ms: Minimum duration to consider as loop.

    Returns:
        True if likely a loop, False otherwise.
    """
    info = get_audio_info(file_path)

    # Short samples are not loops
    if info["duration_ms"] < min_duration_ms:
        return False

    # Could add more sophisticated detection here:
    # - Check for repetitive patterns
    # - Look for seamless loop points
    # For now, just use duration heuristic
    return info["duration_ms"] >= min_duration_ms


def analyze_sample(file_path: str | Path) -> dict[str, Any]:
    """Perform full analysis on an audio sample.

    Combines audio info, BPM detection, key detection, and spectral analysis.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with all analysis results.
    """
    # Get basic info
    result = get_audio_info(file_path)

    # Detect BPM
    bpm, bpm_confidence = detect_bpm(file_path)
    result["bpm"] = bpm
    result["bpm_confidence"] = bpm_confidence

    # Detect key
    key, key_confidence = detect_key(file_path)
    result["detected_key"] = key
    result["key_confidence"] = key_confidence

    # Extract spectral features
    spectral = extract_spectral_features(file_path)
    result.update(spectral)

    # Determine if loop
    result["is_loop"] = is_likely_loop(file_path)

    return result
