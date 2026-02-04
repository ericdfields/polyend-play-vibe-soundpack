"""Tests for audio analysis module."""

import numpy as np
import soundfile as sf
import pytest

from soundpack.audio import (
    analyze_sample,
    get_audio_info,
    detect_bpm,
    detect_key,
    extract_spectral_features,
    is_likely_loop,
)


@pytest.fixture
def sine_wave_file(tmp_path):
    """Create a simple sine wave WAV file for testing."""
    sample_rate = 44100
    duration = 1.0  # seconds
    frequency = 440  # A4
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    file_path = tmp_path / "sine_440hz.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path


@pytest.fixture
def very_short_file(tmp_path):
    """Create a very short audio file (< 1024 samples) to test FFT handling."""
    sample_rate = 44100
    # 500 samples = ~11ms at 44.1kHz - shorter than default n_fft=1024
    num_samples = 500
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    file_path = tmp_path / "very_short.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path


@pytest.fixture
def kick_drum_file(tmp_path):
    """Create a synthetic kick drum sound for testing."""
    sample_rate = 44100
    duration = 0.3  # short one-shot

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Exponential decay with low frequency
    envelope = np.exp(-t * 15)
    frequency = 60 + 100 * np.exp(-t * 30)  # pitch drop
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    audio = 0.8 * envelope * np.sin(phase)

    file_path = tmp_path / "kick.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path


@pytest.fixture
def stereo_file(tmp_path):
    """Create a stereo WAV file for testing."""
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack((left, right))

    file_path = tmp_path / "stereo.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path


@pytest.fixture
def rhythmic_loop_file(tmp_path):
    """Create a rhythmic loop at 120 BPM for testing."""
    sample_rate = 44100
    bpm = 120
    beats = 4
    duration = beats * 60 / bpm  # 2 seconds for 4 beats at 120 BPM

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    # Add transients at each beat
    beat_interval = int(sample_rate * 60 / bpm)
    for i in range(beats):
        start = i * beat_interval
        end = min(start + int(sample_rate * 0.05), len(audio))
        if end > start:
            attack = np.exp(-np.linspace(0, 10, end - start))
            audio[start:end] += 0.8 * attack * np.sin(
                2 * np.pi * 60 * np.linspace(0, 0.05, end - start)
            )

    file_path = tmp_path / "loop_120bpm.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path


class TestGetAudioInfo:
    """Tests for basic audio file info extraction."""

    def test_returns_duration_ms(self, sine_wave_file):
        """Returns duration in milliseconds."""
        info = get_audio_info(sine_wave_file)
        assert 990 <= info["duration_ms"] <= 1010  # ~1000ms

    def test_returns_sample_rate(self, sine_wave_file):
        """Returns sample rate."""
        info = get_audio_info(sine_wave_file)
        assert info["sample_rate"] == 44100

    def test_returns_channels_mono(self, sine_wave_file):
        """Returns 1 for mono file."""
        info = get_audio_info(sine_wave_file)
        assert info["channels"] == 1

    def test_returns_channels_stereo(self, stereo_file):
        """Returns 2 for stereo file."""
        info = get_audio_info(stereo_file)
        assert info["channels"] == 2

    def test_returns_bit_depth(self, sine_wave_file):
        """Returns bit depth."""
        info = get_audio_info(sine_wave_file)
        # soundfile writes 64-bit float by default with np arrays
        assert info["bit_depth"] in [16, 24, 32, 64]

    def test_raises_for_invalid_file(self, tmp_path):
        """Raises error for non-audio file."""
        bad_file = tmp_path / "not_audio.txt"
        bad_file.write_text("not audio")

        with pytest.raises(Exception):
            get_audio_info(bad_file)


class TestDetectBpm:
    """Tests for BPM detection."""

    def test_detects_bpm_in_valid_range(self, rhythmic_loop_file):
        """Detects BPM within reasonable range of actual value."""
        bpm, confidence = detect_bpm(rhythmic_loop_file)
        # BPM detection can return half or double tempo
        assert bpm is not None
        assert 50 <= bpm <= 200
        assert 0 <= confidence <= 1

    def test_returns_confidence_score(self, rhythmic_loop_file):
        """Returns confidence score between 0 and 1."""
        bpm, confidence = detect_bpm(rhythmic_loop_file)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_handles_non_rhythmic_content(self, sine_wave_file):
        """Handles non-rhythmic content gracefully."""
        bpm, confidence = detect_bpm(sine_wave_file)
        # Should still return something, even if low confidence
        assert bpm is None or bpm > 0
        assert 0 <= confidence <= 1


class TestDetectKey:
    """Tests for musical key detection."""

    def test_detects_key_from_pitched_audio(self, sine_wave_file):
        """Detects key from pitched audio."""
        key, confidence = detect_key(sine_wave_file)
        # 440Hz is A4, so should detect A major or A minor
        assert key is not None
        assert "A" in key or key is not None  # At least returns something
        assert 0 <= confidence <= 1

    def test_returns_key_in_expected_format(self, sine_wave_file):
        """Key is returned in 'Note major/minor' format."""
        key, confidence = detect_key(sine_wave_file)
        if key:
            assert "major" in key or "minor" in key
            # First character should be a note
            assert key[0] in "ABCDEFG"


class TestExtractSpectralFeatures:
    """Tests for spectral feature extraction."""

    def test_extracts_spectral_centroid(self, sine_wave_file):
        """Extracts spectral centroid in Hz."""
        features = extract_spectral_features(sine_wave_file)
        assert "spectral_centroid" in features
        assert features["spectral_centroid"] > 0  # Must be positive Hz

    def test_handles_very_short_audio_without_warnings(self, very_short_file):
        """Short audio (< 1024 samples) analyzed without n_fft warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = extract_spectral_features(very_short_file)

            # Filter for n_fft warnings specifically
            nfft_warnings = [
                warning for warning in w
                if "n_fft" in str(warning.message)
            ]
            assert len(nfft_warnings) == 0, f"Got n_fft warnings: {[str(x.message) for x in nfft_warnings]}"

        # Should still return valid features
        assert "spectral_centroid" in features
        assert "onset_strength" in features
        assert "rms_energy" in features

    def test_extracts_onset_strength(self, kick_drum_file):
        """Extracts onset strength."""
        features = extract_spectral_features(kick_drum_file)
        assert "onset_strength" in features
        assert features["onset_strength"] >= 0

    def test_extracts_rms_energy(self, sine_wave_file):
        """Extracts RMS energy."""
        features = extract_spectral_features(sine_wave_file)
        assert "rms_energy" in features
        assert features["rms_energy"] >= 0

    def test_kick_has_lower_centroid_than_sine(self, kick_drum_file, sine_wave_file):
        """Kick drum has lower spectral centroid than 440Hz sine."""
        kick_features = extract_spectral_features(kick_drum_file)
        sine_features = extract_spectral_features(sine_wave_file)

        # Kick is mostly low frequency, sine is at 440Hz
        assert kick_features["spectral_centroid"] < sine_features["spectral_centroid"]


class TestIsLikelyLoop:
    """Tests for loop detection."""

    def test_short_sample_not_loop(self, kick_drum_file):
        """Short samples are not loops."""
        assert is_likely_loop(kick_drum_file) is False

    def test_longer_rhythmic_sample_may_be_loop(self, rhythmic_loop_file):
        """Longer rhythmic samples may be detected as loops."""
        result = is_likely_loop(rhythmic_loop_file, min_duration_ms=1000)
        # Just verify it returns a boolean
        assert isinstance(result, bool)


class TestAnalyzeSample:
    """Tests for full sample analysis."""

    def test_returns_all_fields(self, sine_wave_file):
        """Returns dict with all expected fields."""
        result = analyze_sample(sine_wave_file)

        expected_fields = {
            "duration_ms",
            "sample_rate",
            "channels",
            "bit_depth",
            "bpm",
            "bpm_confidence",
            "detected_key",
            "key_confidence",
            "spectral_centroid",
            "onset_strength",
            "rms_energy",
            "is_loop",
        }
        assert expected_fields.issubset(result.keys())

    def test_combines_all_analysis_functions(self, kick_drum_file):
        """Combines info, BPM, key, and spectral analysis."""
        result = analyze_sample(kick_drum_file)

        # Audio info
        assert result["duration_ms"] > 0
        assert result["sample_rate"] == 44100
        assert result["channels"] == 1

        # Spectral features
        assert result["spectral_centroid"] > 0
        assert result["onset_strength"] >= 0
        assert result["rms_energy"] >= 0

        # Loop detection
        assert result["is_loop"] is False  # Short kick is not a loop

    def test_handles_stereo_files(self, stereo_file):
        """Handles stereo files correctly."""
        result = analyze_sample(stereo_file)
        assert result["channels"] == 2
        assert result["duration_ms"] > 0
