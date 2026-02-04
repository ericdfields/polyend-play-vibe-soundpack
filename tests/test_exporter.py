"""Tests for pack exporter module."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundpack.exporter import (
    truncate_filename,
    generate_unique_filename,
    export_pack,
    validate_wav_format,
    FILENAME_MAX_LENGTH,
    MAX_SAMPLES_PER_PACK,
)


class TestTruncateFilename:
    """Tests for filename truncation."""

    def test_short_filename_unchanged(self):
        """Short filenames are not modified."""
        result = truncate_filename("kick.wav")
        assert result == "kick.wav"

    def test_exactly_max_length_unchanged(self):
        """Filename at exactly max length is not modified."""
        # 16 chars + .wav = 20 total
        name = "a" * 16 + ".wav"
        result = truncate_filename(name)
        assert result == name

    def test_long_filename_truncated(self):
        """Long filenames are truncated to max length."""
        long_name = "808_kick_heavy_distorted_dark_punchy_01.wav"
        result = truncate_filename(long_name)
        assert len(Path(result).stem) <= FILENAME_MAX_LENGTH

    def test_preserves_extension(self):
        """Truncation preserves .wav extension."""
        long_name = "very_long_sample_name_that_exceeds_limit.wav"
        result = truncate_filename(long_name)
        assert result.endswith(".wav")

    def test_preserves_trailing_numbers(self):
        """Truncation attempts to preserve trailing numbers."""
        # This test verifies the intelligent truncation strategy
        name = "808_kick_heavy_distorted_01.wav"
        result = truncate_filename(name)
        # Should keep the 01 suffix
        assert "01" in result or result.endswith("01.wav")

    def test_handles_no_extension(self):
        """Handles filenames without extension."""
        result = truncate_filename("kick_drum_sample", preserve_extension=False)
        assert len(result) <= FILENAME_MAX_LENGTH


class TestGenerateUniqueFilename:
    """Tests for unique filename generation."""

    def test_unique_name_unchanged(self):
        """Unique names are returned unchanged."""
        existing = {"kick.wav", "snare.wav"}
        result = generate_unique_filename("hihat.wav", existing)
        assert result == "hihat.wav"

    def test_adds_suffix_for_collision(self):
        """Adds numeric suffix when name collides."""
        existing = {"kick.wav"}
        result = generate_unique_filename("kick.wav", existing)
        assert result != "kick.wav"
        assert "kick" in result
        assert result.endswith(".wav")

    def test_increments_suffix_for_multiple_collisions(self):
        """Increments suffix for multiple collisions."""
        existing = {"kick.wav", "kick_1.wav", "kick_2.wav"}
        result = generate_unique_filename("kick.wav", existing)
        assert result not in existing
        assert "kick" in result

    def test_truncates_then_makes_unique(self):
        """Long names are truncated before making unique."""
        long_name = "very_long_sample_name_that_exceeds_limit.wav"
        existing = set()
        result = generate_unique_filename(long_name, existing)
        assert len(Path(result).stem) <= FILENAME_MAX_LENGTH


class TestValidateWavFormat:
    """Tests for WAV format validation."""

    @pytest.fixture
    def valid_wav(self, tmp_path):
        """Create a valid WAV file."""
        audio = np.zeros(44100, dtype=np.int16)
        path = tmp_path / "valid.wav"
        sf.write(path, audio, 44100, subtype="PCM_16")
        return path

    @pytest.fixture
    def stereo_wav(self, tmp_path):
        """Create a valid stereo WAV file."""
        audio = np.zeros((44100, 2), dtype=np.int16)
        path = tmp_path / "stereo.wav"
        sf.write(path, audio, 44100, subtype="PCM_16")
        return path

    @pytest.fixture
    def wrong_sample_rate(self, tmp_path):
        """Create WAV with wrong sample rate."""
        audio = np.zeros(48000, dtype=np.int16)
        path = tmp_path / "48k.wav"
        sf.write(path, audio, 48000, subtype="PCM_16")
        return path

    def test_valid_mono_wav(self, valid_wav):
        """Accepts valid mono 44.1kHz WAV."""
        is_valid, errors = validate_wav_format(valid_wav)
        assert is_valid
        assert len(errors) == 0

    def test_valid_stereo_wav(self, stereo_wav):
        """Accepts valid stereo 44.1kHz WAV."""
        is_valid, errors = validate_wav_format(stereo_wav)
        assert is_valid
        assert len(errors) == 0

    def test_rejects_wrong_sample_rate(self, wrong_sample_rate):
        """Rejects WAV with wrong sample rate."""
        is_valid, errors = validate_wav_format(wrong_sample_rate)
        assert not is_valid
        assert any("sample rate" in e.lower() for e in errors)

    def test_returns_errors_list(self, wrong_sample_rate):
        """Returns list of validation errors."""
        is_valid, errors = validate_wav_format(wrong_sample_rate)
        assert isinstance(errors, list)
        assert len(errors) > 0


class TestExportPack:
    """Tests for full pack export."""

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample WAV files for testing."""
        files = []
        for name in ["kick.wav", "snare.wav", "hihat.wav"]:
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = tmp_path / "source" / name
            path.parent.mkdir(exist_ok=True)
            sf.write(path, audio, 44100)
            files.append(path)
        return files

    def test_creates_output_directory(self, sample_files, tmp_path):
        """Export creates output directory."""
        output_dir = tmp_path / "output" / "MyPack"
        samples = [{"file_path": str(f), "filename": f.name} for f in sample_files]

        export_pack(samples, output_dir, "MyPack")

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_copies_all_samples(self, sample_files, tmp_path):
        """Export copies all sample files."""
        output_dir = tmp_path / "output" / "MyPack"
        samples = [{"file_path": str(f), "filename": f.name} for f in sample_files]

        result = export_pack(samples, output_dir, "MyPack")

        assert len(list(output_dir.glob("*.wav"))) == 3
        assert result["exported_count"] == 3

    def test_returns_filename_mapping(self, sample_files, tmp_path):
        """Export returns mapping of original to exported filenames."""
        output_dir = tmp_path / "output" / "MyPack"
        samples = [{"file_path": str(f), "filename": f.name} for f in sample_files]

        result = export_pack(samples, output_dir, "MyPack")

        assert "file_mapping" in result
        assert len(result["file_mapping"]) == 3

    def test_truncates_long_filenames(self, tmp_path):
        """Export truncates long filenames."""
        # Create file with long name
        long_name = "808_kick_heavy_distorted_dark_punchy_layered_01.wav"
        audio = np.random.randn(22050).astype(np.float32) * 0.5
        source_path = tmp_path / "source" / long_name
        source_path.parent.mkdir(exist_ok=True)
        sf.write(source_path, audio, 44100)

        output_dir = tmp_path / "output" / "MyPack"
        samples = [{"file_path": str(source_path), "filename": long_name}]

        result = export_pack(samples, output_dir, "MyPack")

        # Check exported filename is truncated
        exported_files = list(output_dir.glob("*.wav"))
        assert len(exported_files) == 1
        assert len(exported_files[0].stem) <= FILENAME_MAX_LENGTH

    def test_handles_duplicate_filenames(self, tmp_path):
        """Export handles duplicate filenames after truncation."""
        # Create two files that would have same truncated name
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        files = []
        for i, suffix in enumerate(["dark", "light"]):
            name = f"very_long_kick_name_{suffix}.wav"
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            files.append({"file_path": str(path), "filename": name})

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(files, output_dir, "MyPack")

        # Both files should be exported with unique names
        exported_files = list(output_dir.glob("*.wav"))
        assert len(exported_files) == 2
        names = {f.name for f in exported_files}
        assert len(names) == 2  # All unique

    def test_enforces_max_samples(self, tmp_path):
        """Export enforces maximum samples per pack."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create more files than allowed
        samples = []
        for i in range(MAX_SAMPLES_PER_PACK + 10):
            name = f"sample_{i:03d}.wav"
            audio = np.zeros(1000, dtype=np.float32)
            path = source_dir / name
            sf.write(path, audio, 44100)
            samples.append({"file_path": str(path), "filename": name})

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(samples, output_dir, "MyPack")

        # Should only export up to max
        assert result["exported_count"] <= MAX_SAMPLES_PER_PACK
        assert result["skipped_count"] >= 10

    def test_organizes_by_instrument_folders(self, tmp_path):
        """Export organizes samples into instrument subfolders."""
        from soundpack.exporter import INSTRUMENT_FOLDERS

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create samples with different tags
        samples = []
        tag_mapping = {}
        for name, tags in [
            ("kick_01.wav", ["kick"]),
            ("snare_01.wav", ["snare"]),
            ("hihat_01.wav", ["hihat"]),
        ]:
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            sample = {"file_path": str(path), "filename": name, "id": len(samples)}
            samples.append(sample)
            tag_mapping[sample["id"]] = tags

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        # Should have subfolders with Play+ compatible names
        assert (output_dir / "Kick").exists()
        assert (output_dir / "Snare").exists()
        assert (output_dir / "HiHat").exists()  # hihat -> HiHat folder
        assert (output_dir / "Kick" / "kick_01.wav").exists()

    def test_maps_hihat_clap_tom_to_folders(self, tmp_path):
        """Maps similar instruments to correct folders."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        samples = []
        tag_mapping = {}
        for name, tags in [
            ("clap_01.wav", ["clap"]),
            ("tom_01.wav", ["tom"]),
            ("closed_hat.wav", ["hihat"]),
            ("rim_01.wav", ["rim"]),
        ]:
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            sample = {"file_path": str(path), "filename": name, "id": len(samples)}
            samples.append(sample)
            tag_mapping[sample["id"]] = tags

        output_dir = tmp_path / "output" / "MyPack"
        export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        # clap/rim -> Snare, tom -> Perc, hihat -> HiHat
        assert (output_dir / "Snare" / "clap_01.wav").exists()
        assert (output_dir / "Perc" / "tom_01.wav").exists()
        assert (output_dir / "HiHat" / "closed_hat.wav").exists()
        assert (output_dir / "Snare" / "rim_01.wav").exists()

    def test_untagged_samples_go_to_fx(self, tmp_path):
        """Samples without matching tags go to FX folder."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        audio = np.random.randn(22050).astype(np.float32) * 0.5
        path = source_dir / "mystery.wav"
        sf.write(path, audio, 44100)

        samples = [{"file_path": str(path), "filename": "mystery.wav", "id": 0}]
        tag_mapping = {0: ["dark", "ambient"]}  # No instrument tags

        output_dir = tmp_path / "output" / "MyPack"
        export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        assert (output_dir / "FX" / "mystery.wav").exists()

    def test_no_tag_mapping_uses_flat_structure(self, sample_files, tmp_path):
        """Without tag_mapping, export uses flat folder structure."""
        output_dir = tmp_path / "output" / "MyPack"
        samples = [{"file_path": str(f), "filename": f.name} for f in sample_files]

        export_pack(samples, output_dir, "MyPack")

        # No subdirectories when no tag_mapping provided
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 0

    def test_warns_about_insufficient_percussion_samples(self, tmp_path):
        """Warns when percussion folders have fewer than 5 samples."""
        from soundpack.exporter import MIN_SAMPLES_PER_PERC_FOLDER

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create only 2 kicks (less than required 5)
        samples = []
        tag_mapping = {}
        for i in range(2):
            name = f"kick_{i:02d}.wav"
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            sample = {"file_path": str(path), "filename": name, "id": i}
            samples.append(sample)
            tag_mapping[i] = ["kick"]

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        # Should have warning about Kick folder
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert any("Kick" in w and "Beat Fill" in w for w in result["warnings"])

    def test_no_warning_when_enough_percussion_samples(self, tmp_path):
        """No warning when percussion folders have 5+ samples."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create 5 kicks (meets requirement)
        samples = []
        tag_mapping = {}
        for i in range(5):
            name = f"kick_{i:02d}.wav"
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            sample = {"file_path": str(path), "filename": name, "id": i}
            samples.append(sample)
            tag_mapping[i] = ["kick"]

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        # Should have no warnings about Kick folder
        kick_warnings = [w for w in result.get("warnings", []) if "Kick" in w]
        assert len(kick_warnings) == 0

    def test_returns_folder_counts(self, tmp_path):
        """Export returns count of samples per folder."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        samples = []
        tag_mapping = {}
        for i, (name, tags) in enumerate([
            ("kick_01.wav", ["kick"]),
            ("kick_02.wav", ["kick"]),
            ("snare_01.wav", ["snare"]),
        ]):
            audio = np.random.randn(22050).astype(np.float32) * 0.5
            path = source_dir / name
            sf.write(path, audio, 44100)
            sample = {"file_path": str(path), "filename": name, "id": i}
            samples.append(sample)
            tag_mapping[i] = tags

        output_dir = tmp_path / "output" / "MyPack"
        result = export_pack(samples, output_dir, "MyPack", tag_mapping=tag_mapping)

        assert "folder_counts" in result
        assert result["folder_counts"]["Kick"] == 2
        assert result["folder_counts"]["Snare"] == 1
