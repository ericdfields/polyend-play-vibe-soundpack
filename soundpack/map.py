"""Spectral map: 2D visualization of sample library based on audio features."""

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np

# Version for tracking when to recompute embeddings
MAP_VERSION = 1


def _get_safe_n_fft(signal_length: int, default: int = 2048) -> int:
    """Get an n_fft value that fits the signal length."""
    if signal_length >= default:
        return default
    n_fft = 1
    while n_fft * 2 <= signal_length:
        n_fft *= 2
    return max(64, n_fft)


def extract_map_features(file_path: str | Path) -> dict[str, Any]:
    """Extract comprehensive audio features for spectral map embedding.

    Extracts a rich set of timbral, spectral, and temporal features that
    capture the sonic character of a sample for visualization.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with feature arrays and scalars for embedding.
    """
    y, sr = librosa.load(file_path, sr=22050, mono=True)  # Standardize sample rate

    n_fft = _get_safe_n_fft(len(y))
    hop_length = n_fft // 4

    features: dict[str, Any] = {}

    # === Timbral Features (MFCCs) ===
    # Mel-frequency cepstral coefficients capture timbral texture
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    features["mfcc_mean"] = mfcc.mean(axis=1).tolist()  # 13 coefficients
    features["mfcc_std"] = mfcc.std(axis=1).tolist()  # 13 coefficients

    # === Spectral Features ===
    # Spectral centroid - "brightness"
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    features["spectral_centroid_mean"] = float(spectral_centroid.mean())
    features["spectral_centroid_std"] = float(spectral_centroid.std())

    # Spectral rolloff - frequency below which 85% of energy is contained
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    features["spectral_rolloff_mean"] = float(spectral_rolloff.mean())

    # Spectral bandwidth - width of the spectrum
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    features["spectral_bandwidth_mean"] = float(spectral_bandwidth.mean())

    # Spectral contrast - difference between peaks and valleys in spectrum
    try:
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features["spectral_contrast_mean"] = spectral_contrast.mean(axis=1).tolist()  # 7 bands
    except Exception:
        features["spectral_contrast_mean"] = [0.0] * 7

    # Spectral flatness - how noise-like vs tonal
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    features["spectral_flatness_mean"] = float(spectral_flatness.mean())

    # === Energy/Dynamics Features ===
    # RMS energy - perceived loudness
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    features["rms_mean"] = float(rms.mean())
    features["rms_std"] = float(rms.std())

    # Zero crossing rate - noisiness/percussiveness indicator
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    features["zcr_mean"] = float(zcr.mean())

    # === Temporal Features ===
    # Onset strength - transient sharpness
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features["onset_strength_mean"] = float(onset_env.mean())
    features["onset_strength_max"] = float(onset_env.max())

    # Duration
    features["duration_seconds"] = float(librosa.get_duration(y=y, sr=sr))

    # Attack time estimate (time to reach max amplitude)
    if len(y) > 0:
        max_idx = np.argmax(np.abs(y))
        features["attack_time"] = float(max_idx / sr)
    else:
        features["attack_time"] = 0.0

    # === Rhythm Features (for longer samples) ===
    if len(y) > sr:  # Only for samples > 1 second
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            if hasattr(tempo, "__len__"):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
            features["tempo"] = float(tempo)
        except Exception:
            features["tempo"] = 0.0
    else:
        features["tempo"] = 0.0

    return features


def features_to_vector(features: dict[str, Any]) -> np.ndarray:
    """Convert feature dict to a flat numpy array for embedding.

    The vector is normalized and ready for dimensionality reduction.

    Args:
        features: Feature dict from extract_map_features.

    Returns:
        1D numpy array of features.
    """
    vector = []

    # MFCCs (26 values: 13 mean + 13 std)
    vector.extend(features.get("mfcc_mean", [0.0] * 13))
    vector.extend(features.get("mfcc_std", [0.0] * 13))

    # Spectral features (6 values)
    vector.append(features.get("spectral_centroid_mean", 0.0))
    vector.append(features.get("spectral_centroid_std", 0.0))
    vector.append(features.get("spectral_rolloff_mean", 0.0))
    vector.append(features.get("spectral_bandwidth_mean", 0.0))
    vector.append(features.get("spectral_flatness_mean", 0.0))

    # Spectral contrast (7 values)
    vector.extend(features.get("spectral_contrast_mean", [0.0] * 7))

    # Energy features (3 values)
    vector.append(features.get("rms_mean", 0.0))
    vector.append(features.get("rms_std", 0.0))
    vector.append(features.get("zcr_mean", 0.0))

    # Temporal features (4 values)
    vector.append(features.get("onset_strength_mean", 0.0))
    vector.append(features.get("onset_strength_max", 0.0))
    vector.append(features.get("duration_seconds", 0.0))
    vector.append(features.get("attack_time", 0.0))

    # Rhythm (1 value)
    vector.append(features.get("tempo", 0.0))

    return np.array(vector, dtype=np.float32)


def compute_embeddings(
    feature_vectors: np.ndarray,
    algorithm: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D embeddings from feature vectors using dimensionality reduction.

    Args:
        feature_vectors: 2D array of shape (n_samples, n_features).
        algorithm: 'umap' or 'tsne'.
        n_neighbors: UMAP parameter - larger values = more global structure.
        min_dist: UMAP parameter - minimum distance between points.
        perplexity: t-SNE parameter - related to number of nearest neighbors.
        random_state: Random seed for reproducibility.

    Returns:
        2D array of shape (n_samples, 2) with x, y positions.
    """
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)

    # Handle NaN/Inf values
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = scaled_features.shape[0]

    if n_samples < 2:
        # Can't compute embeddings with less than 2 samples
        return np.zeros((n_samples, 2))

    if algorithm == "umap":
        try:
            import umap

            # Adjust n_neighbors for small datasets
            effective_neighbors = min(n_neighbors, n_samples - 1)
            effective_neighbors = max(2, effective_neighbors)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=effective_neighbors,
                min_dist=min_dist,
                metric="euclidean",
                random_state=random_state,
            )
            embeddings = reducer.fit_transform(scaled_features)
        except ImportError:
            # Fall back to t-SNE if UMAP not installed
            algorithm = "tsne"

    if algorithm == "tsne":
        from sklearn.manifold import TSNE

        # Adjust perplexity for small datasets
        effective_perplexity = min(perplexity, (n_samples - 1) // 3)
        effective_perplexity = max(1, effective_perplexity)

        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=random_state,
            max_iter=1000,
        )
        embeddings = reducer.fit_transform(scaled_features)

    # Normalize to 0-1 range for consistent visualization
    embeddings = normalize_positions(embeddings)

    return embeddings


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize positions to 0-1 range with some padding.

    Args:
        positions: 2D array of shape (n_samples, 2).

    Returns:
        Normalized positions in 0-1 range.
    """
    if positions.shape[0] == 0:
        return positions

    # Add small padding (5% on each side)
    padding = 0.05

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    normalized = np.zeros_like(positions)
    normalized[:, 0] = (positions[:, 0] - x_min) / x_range * (1 - 2 * padding) + padding
    normalized[:, 1] = (positions[:, 1] - y_min) / y_range * (1 - 2 * padding) + padding

    return normalized


def interpret_position(x: float, y: float) -> dict[str, str]:
    """Interpret a map position in human-readable terms.

    The axes tend to correlate with:
    - X axis: soft/ambient (left) to punchy/aggressive (right)
    - Y axis: dark/bass-heavy (bottom) to bright/high-frequency (top)

    Args:
        x: X position (0-1).
        y: Y position (0-1).

    Returns:
        Dict with interpretations for brightness and energy.
    """
    # Brightness (Y axis)
    if y < 0.33:
        brightness = "dark"
    elif y < 0.66:
        brightness = "neutral"
    else:
        brightness = "bright"

    # Energy/Attack (X axis)
    if x < 0.33:
        energy = "soft"
    elif x < 0.66:
        energy = "moderate"
    else:
        energy = "punchy"

    return {"brightness": brightness, "energy": energy}


def find_neighbors(
    sample_idx: int,
    positions: np.ndarray,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Find k nearest neighbors to a sample in the map.

    Args:
        sample_idx: Index of the sample to find neighbors for.
        positions: 2D array of all sample positions.
        k: Number of neighbors to return.

    Returns:
        List of (index, distance) tuples, sorted by distance.
    """
    if sample_idx >= len(positions):
        return []

    target = positions[sample_idx]

    # Compute distances to all other samples
    distances = np.sqrt(np.sum((positions - target) ** 2, axis=1))

    # Get indices sorted by distance (excluding self)
    sorted_indices = np.argsort(distances)
    neighbors = []

    for idx in sorted_indices:
        if idx != sample_idx:
            neighbors.append((int(idx), float(distances[idx])))
            if len(neighbors) >= k:
                break

    return neighbors


def cluster_samples(
    positions: np.ndarray,
    n_clusters: int | None = None,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """Cluster samples based on their map positions.

    Args:
        positions: 2D array of sample positions.
        n_clusters: Number of clusters (auto-detected if None).
        min_cluster_size: Minimum samples per cluster for HDBSCAN.

    Returns:
        Array of cluster labels for each sample.
    """
    from sklearn.cluster import KMeans

    n_samples = positions.shape[0]

    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        # Use sqrt(n) as a reasonable default
        n_clusters = max(2, min(int(np.sqrt(n_samples)), 20))

    # Ensure we don't have more clusters than samples
    n_clusters = min(n_clusters, n_samples)

    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(positions)

    return labels


def export_map_data(
    samples: list[dict[str, Any]],
    positions: np.ndarray,
    output_path: str | Path,
    format: str = "json",
) -> None:
    """Export map data for visualization.

    Args:
        samples: List of sample dicts with id, filename, tags, etc.
        positions: 2D array of positions.
        output_path: Output file path.
        format: 'json' or 'csv'.
    """
    output_path = Path(output_path)

    if format == "json":
        data = []
        for i, sample in enumerate(samples):
            if i < len(positions):
                data.append(
                    {
                        "id": sample.get("id"),
                        "filename": sample.get("filename"),
                        "x": float(positions[i, 0]),
                        "y": float(positions[i, 1]),
                        "tags": sample.get("tags", []),
                        "bpm": sample.get("bpm"),
                        "key": sample.get("detected_key"),
                        "duration_ms": sample.get("duration_ms"),
                    }
                )
        output_path.write_text(json.dumps(data, indent=2))

    elif format == "csv":
        lines = ["id,filename,x,y,tags,bpm,key,duration_ms"]
        for i, sample in enumerate(samples):
            if i < len(positions):
                tags_str = "|".join(sample.get("tags", []))
                lines.append(
                    f"{sample.get('id')},{sample.get('filename')},"
                    f"{positions[i, 0]:.6f},{positions[i, 1]:.6f},"
                    f'"{tags_str}",{sample.get("bpm") or ""},'
                    f'{sample.get("detected_key") or ""},{sample.get("duration_ms") or ""}'
                )
        output_path.write_text("\n".join(lines))
