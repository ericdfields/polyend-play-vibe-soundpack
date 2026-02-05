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


# === Pack Builder Suggestions ===

# Instrument categories for pack building
INSTRUMENT_CATEGORIES = {
    "kick": ["kick", "808", "909"],
    "snare": ["snare", "clap", "rim"],
    "hihat": ["hihat", "hat", "cymbal", "ride"],
    "perc": ["perc", "shaker", "tambourine", "conga", "bongo", "tom"],
    "bass": ["bass", "sub", "808"],
    "synth": ["synth", "pad", "lead", "keys", "piano", "organ"],
    "fx": ["fx", "riser", "impact", "sweep", "noise", "texture"],
    "vocal": ["vocal", "vox", "voice", "choir"],
    "loop": ["loop", "break", "beat"],
}

# Which categories to suggest after each category (the pack building flow)
CATEGORY_FLOW = {
    "kick": ["snare", "hihat"],
    "snare": ["hihat", "perc"],
    "hihat": ["perc", "bass"],
    "perc": ["bass", "synth"],
    "bass": ["synth", "fx"],
    "synth": ["fx", "vocal"],
    "fx": ["vocal", "loop"],
    "vocal": ["loop"],
    "loop": [],
}


def get_sample_category(tags: list[str]) -> str | None:
    """Determine the primary instrument category for a sample based on its tags.

    Args:
        tags: List of tag names.

    Returns:
        Category name or None if not categorizable.
    """
    tags_lower = [t.lower() for t in tags]

    for category, category_tags in INSTRUMENT_CATEGORIES.items():
        for tag in category_tags:
            if tag in tags_lower:
                return category

    return None


def get_pack_centroid(
    pack_samples: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Calculate the centroid position of samples in a pack.

    Args:
        pack_samples: List of sample dicts with map_x, map_y.

    Returns:
        (x, y) centroid or None if no valid positions.
    """
    positions = [
        (s["map_x"], s["map_y"])
        for s in pack_samples
        if s.get("map_x") is not None and s.get("map_y") is not None
    ]

    if not positions:
        return None

    x_mean = sum(p[0] for p in positions) / len(positions)
    y_mean = sum(p[1] for p in positions) / len(positions)

    return (x_mean, y_mean)


def get_common_tags(pack_samples: list[dict[str, Any]], min_frequency: float = 0.3) -> list[str]:
    """Find tags that appear frequently across pack samples.

    Args:
        pack_samples: List of sample dicts with tags.
        min_frequency: Minimum fraction of samples that must have a tag.

    Returns:
        List of common tag names.
    """
    if not pack_samples:
        return []

    tag_counts: dict[str, int] = {}
    for sample in pack_samples:
        for tag in sample.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    threshold = len(pack_samples) * min_frequency
    common = [tag for tag, count in tag_counts.items() if count >= threshold]

    return common


def get_pack_bpm_range(pack_samples: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Get the BPM range of samples in a pack.

    Args:
        pack_samples: List of sample dicts with bpm.

    Returns:
        (min_bpm, max_bpm) or None if no BPM data.
    """
    bpms = [s["bpm"] for s in pack_samples if s.get("bpm")]
    if not bpms:
        return None
    return (min(bpms), max(bpms))


def suggest_complements(
    pack_samples: list[dict[str, Any]],
    all_samples: list[dict[str, Any]],
    target_categories: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Suggest samples that complement the current pack.

    Uses multiple signals to find samples that "work with" what's already selected:
    - Tag coherence: shares genre/mood/character tags
    - Spectral complement: different position (avoids frequency clash)
    - BPM compatibility: similar tempo for loops
    - Category targeting: focuses on needed instrument types

    Args:
        pack_samples: Samples already in the pack.
        all_samples: All available samples to choose from.
        target_categories: Specific categories to suggest (e.g., ["snare", "hihat"]).
        limit: Maximum suggestions to return.

    Returns:
        List of sample dicts with added 'suggestion_score' and 'suggestion_reason'.
    """
    if not pack_samples:
        return []

    # Analyze the current pack
    pack_centroid = get_pack_centroid(pack_samples)
    pack_tags = get_common_tags(pack_samples)
    pack_bpm = get_pack_bpm_range(pack_samples)
    pack_ids = {s["id"] for s in pack_samples}

    # Determine what categories we already have
    pack_categories = set()
    for sample in pack_samples:
        cat = get_sample_category(sample.get("tags", []))
        if cat:
            pack_categories.add(cat)

    # If no target categories specified, suggest next in flow
    if not target_categories:
        target_categories = []
        for cat in pack_categories:
            target_categories.extend(CATEGORY_FLOW.get(cat, []))
        # Remove categories we already have plenty of
        target_categories = [c for c in target_categories if c not in pack_categories]
        # If still empty, suggest common percussion
        if not target_categories:
            target_categories = ["snare", "hihat", "perc"]

    # Score all candidates
    scored_samples = []

    for sample in all_samples:
        # Skip samples already in pack
        if sample["id"] in pack_ids:
            continue

        # Skip samples without map positions
        if sample.get("map_x") is None:
            continue

        sample_tags = sample.get("tags", [])
        sample_category = get_sample_category(sample_tags)

        # Skip if not in target categories
        if target_categories and sample_category not in target_categories:
            continue

        score = 0.0
        reasons = []

        # 1. Tag coherence (same vibe) - up to 30 points
        shared_tags = set(sample_tags) & set(pack_tags)
        # Weight genre/mood/character tags higher than instrument tags
        vibe_tags = [t for t in shared_tags if t not in sum(INSTRUMENT_CATEGORIES.values(), [])]
        tag_score = len(vibe_tags) * 10 + len(shared_tags) * 2
        tag_score = min(tag_score, 30)
        if tag_score > 0:
            score += tag_score
            reasons.append(f"shares {', '.join(list(shared_tags)[:3])}")

        # 2. Spectral complement - up to 25 points
        # We want samples that are DIFFERENT enough to not clash,
        # but not SO different they don't fit
        if pack_centroid:
            sample_pos = (sample["map_x"], sample["map_y"])
            distance = np.sqrt(
                (sample_pos[0] - pack_centroid[0]) ** 2 +
                (sample_pos[1] - pack_centroid[1]) ** 2
            )
            # Sweet spot: 0.15 to 0.45 distance
            if 0.15 <= distance <= 0.45:
                complement_score = 25
                reasons.append("good spectral balance")
            elif 0.08 <= distance <= 0.6:
                complement_score = 15
            else:
                complement_score = 5
            score += complement_score

        # 3. BPM compatibility - up to 20 points
        if pack_bpm and sample.get("bpm"):
            bpm_diff = min(
                abs(sample["bpm"] - pack_bpm[0]),
                abs(sample["bpm"] - pack_bpm[1])
            )
            if bpm_diff <= 2:
                score += 20
                reasons.append("BPM match")
            elif bpm_diff <= 5:
                score += 15
            elif bpm_diff <= 10:
                score += 10

        # 4. Category bonus - up to 15 points
        if sample_category in target_categories[:2]:
            score += 15
            reasons.append(f"fills {sample_category} slot")
        elif sample_category in target_categories:
            score += 10

        # 5. Brightness/energy complement - up to 10 points
        # If pack is dark (low y), prefer brighter complements for contrast
        if pack_centroid:
            pack_y = pack_centroid[1]
            sample_y = sample["map_y"]
            # Mild preference for samples on opposite brightness
            if pack_y < 0.4 and sample_y > 0.5:
                score += 10
                reasons.append("adds brightness")
            elif pack_y > 0.6 and sample_y < 0.5:
                score += 10
                reasons.append("adds depth")

        if score > 0:
            result = dict(sample)
            result["suggestion_score"] = score
            result["suggestion_reason"] = "; ".join(reasons) if reasons else "potential match"
            scored_samples.append(result)

    # Sort by score and return top suggestions
    scored_samples.sort(key=lambda x: x["suggestion_score"], reverse=True)

    return scored_samples[:limit]


def get_next_categories(pack_samples: list[dict[str, Any]]) -> list[str]:
    """Determine what instrument categories to suggest next.

    Args:
        pack_samples: Current pack samples.

    Returns:
        List of suggested category names.
    """
    if not pack_samples:
        return ["kick", "snare", "hihat"]

    # What do we have?
    pack_categories = set()
    for sample in pack_samples:
        cat = get_sample_category(sample.get("tags", []))
        if cat:
            pack_categories.add(cat)

    # What should come next?
    next_cats = []
    for cat in pack_categories:
        for next_cat in CATEGORY_FLOW.get(cat, []):
            if next_cat not in pack_categories and next_cat not in next_cats:
                next_cats.append(next_cat)

    # If nothing suggested, recommend basic percussion
    if not next_cats:
        for cat in ["kick", "snare", "hihat", "bass"]:
            if cat not in pack_categories:
                next_cats.append(cat)

    return next_cats[:3]


def analyze_pack_balance(
    pack_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze the spectral and categorical balance of a pack.

    Args:
        pack_samples: Current pack samples.

    Returns:
        Dict with balance analysis.
    """
    if not pack_samples:
        return {"empty": True}

    # Category distribution
    categories: dict[str, int] = {}
    for sample in pack_samples:
        cat = get_sample_category(sample.get("tags", [])) or "other"
        categories[cat] = categories.get(cat, 0) + 1

    # Spectral distribution (quadrants)
    quadrants = {"dark_soft": 0, "dark_punchy": 0, "bright_soft": 0, "bright_punchy": 0}
    for sample in pack_samples:
        if sample.get("map_x") is None:
            continue
        x, y = sample["map_x"], sample["map_y"]
        if y < 0.5:
            if x < 0.5:
                quadrants["dark_soft"] += 1
            else:
                quadrants["dark_punchy"] += 1
        else:
            if x < 0.5:
                quadrants["bright_soft"] += 1
            else:
                quadrants["bright_punchy"] += 1

    # Centroid
    centroid = get_pack_centroid(pack_samples)

    # Common tags
    common_tags = get_common_tags(pack_samples)

    # Suggestions for balance
    suggestions = []
    total = len(pack_samples)

    # Check percussion balance
    perc_count = categories.get("kick", 0) + categories.get("snare", 0) + categories.get("hihat", 0)
    if perc_count < total * 0.3 and total > 5:
        suggestions.append("Consider adding more percussion")

    # Check spectral balance
    total_positioned = sum(quadrants.values())
    if total_positioned > 0:
        dominant = max(quadrants, key=quadrants.get)
        if quadrants[dominant] > total_positioned * 0.6:
            opposite = {
                "dark_soft": "bright_punchy",
                "dark_punchy": "bright_soft",
                "bright_soft": "dark_punchy",
                "bright_punchy": "dark_soft",
            }[dominant]
            suggestions.append(f"Pack leans {dominant.replace('_', ' ')}; consider {opposite.replace('_', ' ')} samples")

    return {
        "total": total,
        "categories": categories,
        "spectral_quadrants": quadrants,
        "centroid": centroid,
        "common_tags": common_tags,
        "next_categories": get_next_categories(pack_samples),
        "suggestions": suggestions,
    }


# Default target sizes for each category in a balanced pack
AUTOFILL_TARGETS = {
    "kick": 4,
    "snare": 4,
    "hihat": 4,
    "perc": 3,
    "bass": 3,
    "synth": 2,
    "fx": 2,
    "vocal": 1,
}


def smart_autofill(
    pack_samples: list[dict[str, Any]],
    all_samples: list[dict[str, Any]],
    target_count: int = 64,
    max_size_mb: float = 30.0,
    preserve_vibe: bool = True,
) -> list[dict[str, Any]]:
    """Automatically fill out a pack to create a balanced, export-ready collection.

    Takes the current pack selection and intelligently adds complementary samples
    to create a well-rounded pack with good category distribution. Uses a hybrid
    approach: targets a sample count but stops early if hitting the size limit.

    Args:
        pack_samples: Samples already in the pack.
        all_samples: All available samples to choose from.
        target_count: Target total sample count (default 64).
        max_size_mb: Maximum pack size in MB (default 30, leaving buffer for 32MB limit).
        preserve_vibe: If True, strongly prefer samples matching pack's vibe.

    Returns:
        List of samples to ADD to the pack (not including existing samples).
    """
    if not pack_samples:
        # No starting point - return empty
        return []

    # Calculate current pack size in bytes
    current_size_bytes = sum(
        s.get("file_size_bytes", 0) or 0 for s in pack_samples
    )
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    # Calculate how many more samples we could add (by count)
    samples_to_add = target_count - len(pack_samples)
    if samples_to_add <= 0:
        return []

    # Check if we're already at or near size limit
    if current_size_bytes >= max_size_bytes:
        return []

    pack_ids = {s["id"] for s in pack_samples}

    # Analyze current pack composition
    current_categories: dict[str, int] = {}
    for sample in pack_samples:
        cat = get_sample_category(sample.get("tags", [])) or "other"
        current_categories[cat] = current_categories.get(cat, 0) + 1

    # Get pack characteristics for matching
    pack_tags = get_common_tags(pack_samples, min_frequency=0.2)
    pack_centroid = get_pack_centroid(pack_samples)
    pack_bpm = get_pack_bpm_range(pack_samples)

    # Calculate target count for each category based on target_count
    scale_factor = target_count / 32.0  # Base targets are for 32 samples
    category_targets = {
        cat: max(1, int(count * scale_factor))
        for cat, count in AUTOFILL_TARGETS.items()
    }

    # Calculate how many we need for each category
    category_needs: dict[str, int] = {}
    for cat, target in category_targets.items():
        current = current_categories.get(cat, 0)
        if current < target:
            category_needs[cat] = target - current

    # Priority order for filling categories
    priority_order = ["kick", "snare", "hihat", "perc", "bass", "synth", "fx", "vocal"]

    # Filter out samples already in pack and without map data
    available = [
        s for s in all_samples
        if s["id"] not in pack_ids and s.get("map_x") is not None
    ]

    # Score all available samples for vibe matching
    def score_sample(sample: dict[str, Any]) -> float:
        """Score how well a sample matches the pack's vibe."""
        score = 0.0
        sample_tags = sample.get("tags", [])

        # Tag overlap (vibe matching)
        if preserve_vibe and pack_tags:
            shared = set(sample_tags) & set(pack_tags)
            # Weight non-instrument tags higher
            vibe_tags = [
                t for t in shared
                if t not in sum(INSTRUMENT_CATEGORIES.values(), [])
            ]
            score += len(vibe_tags) * 15 + len(shared) * 3

        # Spectral proximity to pack centroid
        if pack_centroid:
            distance = np.sqrt(
                (sample["map_x"] - pack_centroid[0]) ** 2 +
                (sample["map_y"] - pack_centroid[1]) ** 2
            )
            # Prefer samples within reasonable distance but not identical
            if 0.1 <= distance <= 0.4:
                score += 20
            elif distance <= 0.6:
                score += 10

        # BPM compatibility
        if pack_bpm and sample.get("bpm"):
            bpm_diff = min(
                abs(sample["bpm"] - pack_bpm[0]),
                abs(sample["bpm"] - pack_bpm[1])
            )
            if bpm_diff <= 3:
                score += 15
            elif bpm_diff <= 8:
                score += 8

        return score

    # Pre-score all available samples
    for sample in available:
        sample["_vibe_score"] = score_sample(sample)

    # Collect samples to add
    samples_to_return: list[dict[str, Any]] = []
    added_ids: set[int] = set()
    running_size_bytes = current_size_bytes

    # Track how many we've added per category (including existing pack samples)
    category_counts: dict[str, int] = dict(current_categories)

    # Maximum samples per category (150% of target, minimum target + 2)
    category_max: dict[str, int] = {
        cat: max(int(count * 1.5), count + 2)
        for cat, count in category_targets.items()
    }

    def can_add_sample(sample: dict[str, Any]) -> bool:
        """Check if adding this sample would exceed size limit."""
        sample_size = sample.get("file_size_bytes", 0) or 0
        return (running_size_bytes + sample_size) <= max_size_bytes

    # Fill each category in priority order
    for category in priority_order:
        needed = category_needs.get(category, 0)
        if needed <= 0:
            continue

        # Find samples in this category
        candidates = [
            s for s in available
            if get_sample_category(s.get("tags", [])) == category
            and s["id"] not in added_ids
        ]

        # Sort by vibe score
        candidates.sort(key=lambda x: x.get("_vibe_score", 0), reverse=True)

        # Take top candidates (checking size limit)
        added_in_category = 0
        for sample in candidates:
            if added_in_category >= needed:
                break
            if not can_add_sample(sample):
                continue  # Skip this sample, try next one

            sample_size = sample.get("file_size_bytes", 0) or 0
            result = {k: v for k, v in sample.items() if not k.startswith("_")}
            result["autofill_reason"] = f"fills {category} ({len(samples_to_return) + 1}/{samples_to_add})"
            samples_to_return.append(result)
            added_ids.add(sample["id"])
            running_size_bytes += sample_size
            category_counts[category] = category_counts.get(category, 0) + 1
            added_in_category += 1

            if len(samples_to_return) >= samples_to_add:
                break

        if len(samples_to_return) >= samples_to_add:
            break

    # If we still need more samples, fill with best vibe matches BUT respect category caps
    if len(samples_to_return) < samples_to_add:
        remaining = [
            s for s in available
            if s["id"] not in added_ids
        ]
        remaining.sort(key=lambda x: x.get("_vibe_score", 0), reverse=True)

        for sample in remaining:
            if len(samples_to_return) >= samples_to_add:
                break
            if not can_add_sample(sample):
                continue  # Skip this sample, would exceed size limit

            # Check if this category is at its cap
            cat = get_sample_category(sample.get("tags", [])) or "other"
            current_count = category_counts.get(cat, 0)
            max_count = category_max.get(cat, 6)  # default max of 6 for unknown categories

            if current_count >= max_count:
                # Skip this sample, category is full
                continue

            sample_size = sample.get("file_size_bytes", 0) or 0
            result = {k: v for k, v in sample.items() if not k.startswith("_")}
            result["autofill_reason"] = f"vibe match ({cat})"
            samples_to_return.append(result)
            added_ids.add(sample["id"])
            running_size_bytes += sample_size
            category_counts[cat] = current_count + 1

    return samples_to_return
