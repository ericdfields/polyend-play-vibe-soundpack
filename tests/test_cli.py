"""Tests for CLI commands."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from soundpack.cli import cli, parse_id_ranges


class TestParseIdRanges:
    """Tests for ID range parsing."""

    def test_single_ids(self):
        """Parses single IDs."""
        assert parse_id_ranges(("1", "2", "3")) == [1, 2, 3]

    def test_range(self):
        """Parses ranges like 1-5."""
        assert parse_id_ranges(("1-5",)) == [1, 2, 3, 4, 5]

    def test_comma_separated(self):
        """Parses comma-separated values."""
        assert parse_id_ranges(("1,5,10",)) == [1, 5, 10]

    def test_mixed(self):
        """Parses mixed ranges and single values."""
        assert parse_id_ranges(("1-3,10,20-22",)) == [1, 2, 3, 10, 20, 21, 22]

    def test_multiple_args(self):
        """Handles multiple arguments."""
        assert parse_id_ranges(("1-3", "10", "20-22")) == [1, 2, 3, 10, 20, 21, 22]

    def test_ignores_invalid(self):
        """Ignores invalid values."""
        assert parse_id_ranges(("1", "abc", "3")) == [1, 3]


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with sample WAV files."""
    sample_rate = 44100
    duration = 0.5

    # Create kick sample
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    kick = 0.8 * np.exp(-t * 15) * np.sin(2 * np.pi * 60 * t)
    sf.write(tmp_path / "kick_808.wav", kick, sample_rate)

    # Create snare sample
    snare = 0.5 * np.exp(-t * 20) * np.random.randn(len(t))
    sf.write(tmp_path / "snare_punchy.wav", snare, sample_rate)

    # Create nested directory with more samples
    subdir = tmp_path / "drums"
    subdir.mkdir()
    hihat = 0.3 * np.exp(-t * 30) * np.random.randn(len(t))
    sf.write(subdir / "hihat_closed.wav", hihat, sample_rate)

    return tmp_path


@pytest.fixture
def config_and_db(tmp_path):
    """Create temporary config and database paths."""
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "library.db"
    return config_path, db_path


class TestCliBasics:
    """Tests for basic CLI functionality."""

    def test_cli_shows_help(self, runner):
        """CLI shows help without error."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Sound Pack Manager" in result.output or "soundpack" in result.output.lower()

    def test_cli_shows_version(self, runner):
        """CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestImportCommand:
    """Tests for the import command."""

    def test_import_scans_directory(self, runner, sample_dir, config_and_db):
        """Import scans directory and adds samples to database."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                str(sample_dir),
            ],
        )

        assert result.exit_code == 0
        assert "kick_808.wav" in result.output or "2" in result.output  # 2 files in root

    def test_import_recursive(self, runner, sample_dir, config_and_db):
        """Import with --recursive scans subdirectories."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                str(sample_dir),
                "--recursive",
            ],
        )

        assert result.exit_code == 0
        # Should find all 3 files (2 in root + 1 in drums/)
        assert "3" in result.output or "hihat" in result.output.lower()

    def test_import_with_analyze(self, runner, sample_dir, config_and_db):
        """Import with --analyze runs audio analysis."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                str(sample_dir),
                "--analyze",
            ],
        )

        assert result.exit_code == 0
        assert "analyz" in result.output.lower()

    def test_import_nonexistent_directory(self, runner, config_and_db):
        """Import shows error for non-existent directory."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                "/nonexistent/path",
            ],
        )

        assert result.exit_code != 0

    def test_import_skips_large_files(self, runner, config_and_db, tmp_path):
        """Import skips files exceeding max size limit."""
        config_path, db_path = config_and_db

        # Create a "large" file (we'll use a small max-size for testing)
        large_file = tmp_path / "large_sample.wav"
        # Create a 2MB file
        audio = np.zeros(44100 * 20, dtype=np.float32)  # ~20 seconds
        sf.write(large_file, audio, 44100)

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                str(tmp_path),
                "--max-size",
                "1",  # 1 MB limit
            ],
        )

        assert result.exit_code == 0
        assert "exceeding" in result.output.lower() or "limit" in result.output.lower()


class TestListCommand:
    """Tests for the list command."""

    def test_list_shows_samples(self, runner, sample_dir, config_and_db):
        """List shows imported samples."""
        config_path, db_path = config_and_db

        # First import
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )

        # Then list
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "list"],
        )

        assert result.exit_code == 0
        assert "kick" in result.output.lower() or "snare" in result.output.lower()

    def test_list_empty_database(self, runner, config_and_db):
        """List handles empty database gracefully."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "list"],
        )

        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_shows_sample_details(self, runner, sample_dir, config_and_db):
        """Info shows details for a specific sample."""
        config_path, db_path = config_and_db

        # Import first
        runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "import",
                str(sample_dir),
                "--analyze",
            ],
        )

        # Get info by ID
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "info", "1"],
        )

        assert result.exit_code == 0
        # Should show some sample info
        assert ".wav" in result.output.lower() or "duration" in result.output.lower()


class TestTagCommands:
    """Tests for tagging commands."""

    def test_tag_sample(self, runner, sample_dir, config_and_db):
        """Can tag a sample with a tag name."""
        config_path, db_path = config_and_db

        # Import first
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )

        # Tag sample 1 with "kick"
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tag", "1", "kick", "808"],
        )

        assert result.exit_code == 0

    def test_tags_list(self, runner, sample_dir, config_and_db):
        """Tags command lists all tags."""
        config_path, db_path = config_and_db

        # Import and tag
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tag", "1", "kick"],
        )

        # List tags
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tags"],
        )

        assert result.exit_code == 0
        assert "kick" in result.output.lower()

    def test_tags_add(self, runner, config_and_db):
        """Tags add creates new tag in vocabulary."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "tags",
                "add",
                "wobble",
                "--category",
                "character",
            ],
        )

        assert result.exit_code == 0
        assert "wobble" in result.output.lower()


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self, runner, config_and_db):
        """Config show displays current configuration."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "config", "show"],
        )

        assert result.exit_code == 0
        assert "api" in result.output.lower() or "export" in result.output.lower()

    def test_config_set(self, runner, config_and_db):
        """Config set updates configuration value."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "--db",
                str(db_path),
                "config",
                "set",
                "export.max_pack_size",
                "64",
            ],
        )

        assert result.exit_code == 0


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_all(self, runner, sample_dir, config_and_db):
        """Analyze --all analyzes all samples."""
        config_path, db_path = config_and_db

        # Import without analyze
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )

        # Then analyze
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "analyze", "--all"],
        )

        assert result.exit_code == 0
        assert "analyz" in result.output.lower()


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_shows_total_samples(self, runner, sample_dir, config_and_db):
        """Stats shows total sample count."""
        config_path, db_path = config_and_db

        # Import samples
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir), "-r"],
        )

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "stats"],
        )

        assert result.exit_code == 0
        assert "3" in result.output  # 3 samples imported
        assert "sample" in result.output.lower()

    def test_stats_shows_analyzed_count(self, runner, sample_dir, config_and_db):
        """Stats shows how many samples have been analyzed."""
        config_path, db_path = config_and_db

        # Import with analysis
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir), "-a"],
        )

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "stats"],
        )

        assert result.exit_code == 0
        assert "analyzed" in result.output.lower()

    def test_stats_shows_tagged_count(self, runner, sample_dir, config_and_db):
        """Stats shows how many samples have tags."""
        config_path, db_path = config_and_db

        # Import
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )

        # Tag one sample
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tag", "1", "kick"],
        )

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "stats"],
        )

        assert result.exit_code == 0
        assert "tagged" in result.output.lower()

    def test_stats_shows_tag_breakdown(self, runner, sample_dir, config_and_db):
        """Stats shows breakdown by tag category."""
        config_path, db_path = config_and_db

        # Import and tag
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "import", str(sample_dir)],
        )
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tag", "1", "kick"],
        )
        runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "tag", "2", "snare"],
        )

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "stats"],
        )

        assert result.exit_code == 0
        # Should show tag counts
        assert "kick" in result.output.lower() or "instrument" in result.output.lower()

    def test_stats_empty_library(self, runner, config_and_db):
        """Stats works with empty library."""
        config_path, db_path = config_and_db

        result = runner.invoke(
            cli,
            ["--config", str(config_path), "--db", str(db_path), "stats"],
        )

        assert result.exit_code == 0
        assert "0" in result.output
