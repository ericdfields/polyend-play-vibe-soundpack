"""Export packs for Polyend Play Plus."""

import re
import shutil
from pathlib import Path
from typing import Any

import soundfile as sf

# Play Plus constraints
FILENAME_MAX_LENGTH = 16
MAX_SAMPLES_PER_PACK = 255
SUPPORTED_SAMPLE_RATE = 44100
MIN_SAMPLES_PER_PERC_FOLDER = 5  # Required for Beat Fill algorithm
MAX_PACK_SIZE_MB = 32  # Sample pool memory limit

# Instrument folders for organizing samples
# NOTE: Percussion folders MUST contain "Kick", "Snare", or "HiHat" (case sensitive)
# for the Play+ Beat Fill algorithm to recognize them
INSTRUMENT_FOLDERS = ["Kick", "Snare", "HiHat", "Perc", "Synth", "Bass", "FX", "Vocal"]

# Percussion folders that need minimum 5 samples for Beat Fill
PERCUSSION_FOLDERS = ["Kick", "Snare", "HiHat"]

# Map tags to folders
TAG_TO_FOLDER = {
    # Kicks
    "kick": "Kick",
    "808": "Kick",
    # Snares
    "snare": "Snare",
    "clap": "Snare",
    "rim": "Snare",
    "rimshot": "Snare",
    # Hats
    "hihat": "HiHat",
    "hat": "HiHat",
    "cymbal": "HiHat",
    "ride": "HiHat",
    "crash": "HiHat",
    # Percussion
    "perc": "Perc",
    "percussion": "Perc",
    "tom": "Perc",
    "conga": "Perc",
    "bongo": "Perc",
    "shaker": "Perc",
    "tambourine": "Perc",
    # Synth
    "synth": "Synth",
    "pad": "Synth",
    "lead": "Synth",
    "pluck": "Synth",
    "keys": "Synth",
    "piano": "Synth",
    "organ": "Synth",
    "stab": "Synth",
    "chord": "Synth",
    # Bass
    "bass": "Bass",
    "sub": "Bass",
    # Vocal
    "vocal": "Vocal",
    "vox": "Vocal",
    "voice": "Vocal",
    # FX (default)
    "fx": "FX",
    "sfx": "FX",
    "noise": "FX",
    "ambient": "FX",
    "texture": "FX",
    "riser": "FX",
    "sweep": "FX",
    "impact": "FX",
    "loop": "FX",
}


def get_folder_for_tags(tags: list[str]) -> str:
    """Determine the appropriate folder for a sample based on its tags.

    Args:
        tags: List of tag names for the sample.

    Returns:
        Folder name (one of INSTRUMENT_FOLDERS), defaults to "FX".
    """
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in TAG_TO_FOLDER:
            return TAG_TO_FOLDER[tag_lower]
    return "FX"


def truncate_filename(
    filename: str,
    max_length: int = FILENAME_MAX_LENGTH,
    preserve_extension: bool = True,
) -> str:
    """Truncate filename to fit Play Plus display.

    Attempts to preserve:
    - File extension
    - Trailing numbers (e.g., _01, _02)
    - Key identifiers at start

    Args:
        filename: Original filename.
        max_length: Maximum length for the stem (without extension).
        preserve_extension: Whether to preserve the extension.

    Returns:
        Truncated filename.
    """
    path = Path(filename)
    stem = path.stem
    ext = path.suffix if preserve_extension else ""

    if len(stem) <= max_length:
        return stem + ext

    # Try to preserve trailing numbers like _01, _02
    match = re.search(r"[_-](\d{1,3})$", stem)
    if match:
        suffix = match.group(0)
        # Calculate available space for the rest
        available = max_length - len(suffix)
        if available > 0:
            truncated = stem[: available] + suffix
            return truncated + ext

    # Simple truncation
    return stem[:max_length] + ext


def generate_unique_filename(filename: str, existing: set[str]) -> str:
    """Generate a unique filename that doesn't collide with existing names.

    Args:
        filename: Desired filename.
        existing: Set of already-used filenames.

    Returns:
        Unique filename (possibly with numeric suffix).
    """
    # First truncate
    truncated = truncate_filename(filename)

    if truncated not in existing:
        return truncated

    # Add numeric suffix
    path = Path(truncated)
    stem = path.stem
    ext = path.suffix

    counter = 1
    while True:
        suffix = f"_{counter}"
        # Make room for suffix
        available = FILENAME_MAX_LENGTH - len(suffix)
        new_stem = stem[:available] + suffix
        new_name = new_stem + ext

        if new_name not in existing:
            return new_name

        counter += 1
        if counter > 999:
            raise ValueError(f"Could not generate unique name for {filename}")


def validate_wav_format(file_path: str | Path) -> tuple[bool, list[str]]:
    """Validate WAV file format for Play Plus compatibility.

    Args:
        file_path: Path to WAV file.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []

    try:
        info = sf.info(file_path)

        # Check sample rate
        if info.samplerate != SUPPORTED_SAMPLE_RATE:
            errors.append(
                f"Sample rate {info.samplerate} Hz not supported. "
                f"Must be {SUPPORTED_SAMPLE_RATE} Hz."
            )

        # Check format
        if info.format != "WAV":
            errors.append(f"Format {info.format} not supported. Must be WAV.")

        # Check bit depth (accept 16-bit and 24-bit)
        if info.subtype not in ("PCM_16", "PCM_24", "FLOAT"):
            # This is a warning, not an error - most subtypes work
            pass

    except Exception as e:
        errors.append(f"Could not read file: {e}")

    return len(errors) == 0, errors


def export_pack(
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    pack_name: str,
    max_samples: int = MAX_SAMPLES_PER_PACK,
    max_size_mb: float = MAX_PACK_SIZE_MB,
    tag_mapping: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Export samples as a Play Plus compatible pack.

    Args:
        samples: List of sample dicts with file_path, filename, and id keys.
        output_dir: Directory to export to (will be created).
        pack_name: Name of the pack.
        max_samples: Maximum samples to include.
        max_size_mb: Maximum total pack size in MB (Play+ limit is ~32MB).
        tag_mapping: Optional dict mapping sample_id to list of tag names.
                    If provided, samples are organized into instrument subfolders.

    Returns:
        Dict with export results:
        - exported_count: Number of samples exported
        - skipped_count: Number of samples skipped
        - skipped_size: Number skipped due to size limit
        - total_size_mb: Total size of exported pack in MB
        - file_mapping: Dict mapping original filename to exported filename
        - errors: List of error messages
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0
    skipped_size = 0
    total_size_bytes = 0
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    file_mapping: dict[str, str] = {}
    errors: list[str] = []

    # Track existing names per folder to avoid collisions
    existing_names_by_folder: dict[str, set[str]] = {folder: set() for folder in INSTRUMENT_FOLDERS}
    existing_names_flat: set[str] = set()

    for sample in samples:
        if exported_count >= max_samples:
            skipped_count += 1
            continue

        source_path = Path(sample["file_path"])
        original_name = sample.get("filename", source_path.name)

        if not source_path.exists():
            errors.append(f"File not found: {source_path}")
            skipped_count += 1
            continue

        # Check size limit
        file_size = source_path.stat().st_size
        if total_size_bytes + file_size > max_size_bytes:
            skipped_size += 1
            continue

        # Determine destination folder
        if tag_mapping is not None:
            sample_id = sample.get("id")
            tags = tag_mapping.get(sample_id, [])
            folder = get_folder_for_tags(tags)

            # Ensure folder exists
            folder_path = output_path / folder
            folder_path.mkdir(exist_ok=True)

            # Generate unique filename within this folder
            export_name = generate_unique_filename(original_name, existing_names_by_folder[folder])
            existing_names_by_folder[folder].add(export_name)

            dest_path = folder_path / export_name
        else:
            # Flat structure (no tag mapping)
            export_name = generate_unique_filename(original_name, existing_names_flat)
            existing_names_flat.add(export_name)
            dest_path = output_path / export_name

        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            file_mapping[original_name] = export_name
            exported_count += 1
            total_size_bytes += file_size
        except Exception as e:
            errors.append(f"Error copying {original_name}: {e}")
            skipped_count += 1

    # Check percussion folder counts and generate warnings
    warnings: list[str] = []
    folder_counts: dict[str, int] = {}

    if tag_mapping is not None:
        for folder in INSTRUMENT_FOLDERS:
            count = len(existing_names_by_folder.get(folder, set()))
            folder_counts[folder] = count

        # Warn about percussion folders with insufficient samples
        for folder in PERCUSSION_FOLDERS:
            count = folder_counts.get(folder, 0)
            if 0 < count < MIN_SAMPLES_PER_PERC_FOLDER:
                warnings.append(
                    f"{folder} folder has {count} samples (need {MIN_SAMPLES_PER_PERC_FOLDER}+ for Beat Fill)"
                )

    # Add size limit warning if samples were skipped
    if skipped_size > 0:
        warnings.append(
            f"{skipped_size} samples skipped (pack size limit {max_size_mb:.0f} MB reached)"
        )

    total_size_mb = total_size_bytes / (1024 * 1024)

    return {
        "exported_count": exported_count,
        "skipped_count": skipped_count,
        "skipped_size": skipped_size,
        "total_size_mb": total_size_mb,
        "file_mapping": file_mapping,
        "errors": errors,
        "warnings": warnings,
        "folder_counts": folder_counts,
        "output_dir": str(output_path),
        "pack_name": pack_name,
    }
