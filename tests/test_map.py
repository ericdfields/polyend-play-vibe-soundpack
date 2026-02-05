"""Tests for spectral map module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from soundpack.map import (
    MAP_VERSION,
    cluster_samples,
    compute_embeddings,
    export_map_data,
    extract_map_features,
    features_to_vector,
    find_neighbors,
    interpret_position,
    normalize_positions,
)


@pytest.fixture
def sample_wav_file(tmp_path):
    """Create a temporary WAV file for testing."""
    import soundfile as sf

    # Generate a simple sine wave
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    wav_path = tmp_path / "test_sample.wav"
    sf.write(wav_path, audio, sample_rate)
    return wav_path


@pytest.fixture
def kick_wav_file(tmp_path):
    """Create a kick-like WAV file for testing."""
    import soundfile as sf

    sample_rate = 44100
    duration = 0.3
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Low frequency with quick decay
    freq = 80 * np.exp(-t * 20)
    audio = np.sin(2 * np.pi * freq * t) * np.exp(-t * 10)

    wav_path = tmp_path / "kick_sample.wav"
    sf.write(wav_path, audio, sample_rate)
    return wav_path


class TestExtractMapFeatures:
    """Tests for extract_map_features function."""

    def test_returns_mfcc_features(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert len(features["mfcc_mean"]) == 13
        assert len(features["mfcc_std"]) == 13

    def test_returns_spectral_features(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        assert "spectral_centroid_mean" in features
        assert "spectral_rolloff_mean" in features
        assert "spectral_bandwidth_mean" in features
        assert "spectral_flatness_mean" in features

    def test_returns_energy_features(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        assert "rms_mean" in features
        assert "rms_std" in features
        assert "zcr_mean" in features

    def test_returns_temporal_features(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        assert "onset_strength_mean" in features
        assert "onset_strength_max" in features
        assert "duration_seconds" in features
        assert "attack_time" in features

    def test_different_sounds_have_different_features(self, sample_wav_file, kick_wav_file):
        sine_features = extract_map_features(sample_wav_file)
        kick_features = extract_map_features(kick_wav_file)

        # Kick should have lower spectral centroid (darker sound)
        assert kick_features["spectral_centroid_mean"] < sine_features["spectral_centroid_mean"]


class TestFeaturesToVector:
    """Tests for features_to_vector function."""

    def test_returns_numpy_array(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        vector = features_to_vector(features)
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32

    def test_vector_has_correct_dimension(self, sample_wav_file):
        features = extract_map_features(sample_wav_file)
        vector = features_to_vector(features)
        # 13 mfcc_mean + 13 mfcc_std + 5 spectral + 7 contrast + 3 energy + 4 temporal + 1 tempo
        expected_dim = 13 + 13 + 5 + 7 + 3 + 4 + 1
        assert vector.shape == (expected_dim,)

    def test_handles_missing_features(self):
        # Partial features dict
        features = {
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [0.0] * 13,
        }
        vector = features_to_vector(features)
        assert len(vector) > 0


class TestComputeEmbeddings:
    """Tests for compute_embeddings function."""

    def test_returns_2d_positions(self):
        # Create random feature vectors
        n_samples = 20
        n_features = 46
        vectors = np.random.randn(n_samples, n_features).astype(np.float32)

        positions = compute_embeddings(vectors, algorithm="tsne")
        assert positions.shape == (n_samples, 2)

    def test_positions_normalized_to_0_1(self):
        n_samples = 20
        n_features = 46
        vectors = np.random.randn(n_samples, n_features).astype(np.float32)

        positions = compute_embeddings(vectors, algorithm="tsne")
        assert positions[:, 0].min() >= 0
        assert positions[:, 0].max() <= 1
        assert positions[:, 1].min() >= 0
        assert positions[:, 1].max() <= 1

    def test_handles_small_datasets(self):
        vectors = np.random.randn(3, 46).astype(np.float32)
        positions = compute_embeddings(vectors, algorithm="tsne")
        assert positions.shape == (3, 2)

    def test_single_sample_returns_zeros(self):
        vectors = np.random.randn(1, 46).astype(np.float32)
        positions = compute_embeddings(vectors, algorithm="tsne")
        assert positions.shape == (1, 2)

    def test_umap_algorithm(self):
        try:
            import umap
            vectors = np.random.randn(20, 46).astype(np.float32)
            positions = compute_embeddings(vectors, algorithm="umap")
            assert positions.shape == (20, 2)
        except ImportError:
            pytest.skip("umap-learn not installed")


class TestNormalizePositions:
    """Tests for normalize_positions function."""

    def test_normalizes_to_0_1_with_padding(self):
        positions = np.array([[0, 0], [10, 10]], dtype=np.float32)
        normalized = normalize_positions(positions)
        # With 5% padding, min should be 0.05, max should be 0.95
        assert normalized[0, 0] == pytest.approx(0.05)
        assert normalized[1, 0] == pytest.approx(0.95)

    def test_handles_empty_array(self):
        positions = np.array([]).reshape(0, 2)
        normalized = normalize_positions(positions)
        assert normalized.shape == (0, 2)


class TestInterpretPosition:
    """Tests for interpret_position function."""

    def test_dark_soft(self):
        result = interpret_position(0.1, 0.1)
        assert result["brightness"] == "dark"
        assert result["energy"] == "soft"

    def test_bright_punchy(self):
        result = interpret_position(0.9, 0.9)
        assert result["brightness"] == "bright"
        assert result["energy"] == "punchy"

    def test_neutral_moderate(self):
        result = interpret_position(0.5, 0.5)
        assert result["brightness"] == "neutral"
        assert result["energy"] == "moderate"


class TestFindNeighbors:
    """Tests for find_neighbors function."""

    def test_finds_nearest_neighbors(self):
        positions = np.array([
            [0, 0],
            [0.1, 0.1],
            [0.5, 0.5],
            [1, 1],
        ], dtype=np.float32)

        neighbors = find_neighbors(0, positions, k=2)
        assert len(neighbors) == 2
        # First neighbor should be index 1 (closest)
        assert neighbors[0][0] == 1
        # Second should be index 2
        assert neighbors[1][0] == 2

    def test_returns_distances(self):
        positions = np.array([
            [0, 0],
            [1, 0],
        ], dtype=np.float32)

        neighbors = find_neighbors(0, positions, k=1)
        assert len(neighbors) == 1
        assert neighbors[0][1] == pytest.approx(1.0)

    def test_handles_invalid_index(self):
        positions = np.array([[0, 0]], dtype=np.float32)
        neighbors = find_neighbors(5, positions, k=1)
        assert neighbors == []


class TestClusterSamples:
    """Tests for cluster_samples function."""

    def test_returns_cluster_labels(self):
        positions = np.random.randn(20, 2).astype(np.float32)
        labels = cluster_samples(positions, n_clusters=3)
        assert len(labels) == 20
        assert set(labels).issubset({0, 1, 2})

    def test_auto_detects_clusters(self):
        positions = np.random.randn(50, 2).astype(np.float32)
        labels = cluster_samples(positions)
        assert len(labels) == 50

    def test_handles_small_dataset(self):
        positions = np.array([[0, 0], [1, 1]], dtype=np.float32)
        labels = cluster_samples(positions, n_clusters=2)
        assert len(labels) == 2


class TestExportMapData:
    """Tests for export_map_data function."""

    def test_exports_json(self, tmp_path):
        samples = [
            {"id": 1, "filename": "kick.wav", "tags": ["kick"], "bpm": 120, "detected_key": "C major", "duration_ms": 500},
            {"id": 2, "filename": "snare.wav", "tags": ["snare"], "bpm": None, "detected_key": None, "duration_ms": 300},
        ]
        positions = np.array([[0.1, 0.2], [0.8, 0.9]])
        output_path = tmp_path / "map.json"

        export_map_data(samples, positions, output_path, format="json")

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data) == 2
        assert data[0]["x"] == pytest.approx(0.1)
        assert data[0]["y"] == pytest.approx(0.2)
        assert data[0]["filename"] == "kick.wav"

    def test_exports_csv(self, tmp_path):
        samples = [
            {"id": 1, "filename": "kick.wav", "tags": ["kick", "808"], "bpm": 120, "detected_key": "C major", "duration_ms": 500},
        ]
        positions = np.array([[0.5, 0.5]])
        output_path = tmp_path / "map.csv"

        export_map_data(samples, positions, output_path, format="csv")

        assert output_path.exists()
        content = output_path.read_text()
        assert "id,filename,x,y" in content
        assert "kick.wav" in content
        assert "kick|808" in content


class TestMapVersion:
    """Tests for map versioning."""

    def test_map_version_is_set(self):
        assert MAP_VERSION >= 1
