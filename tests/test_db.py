"""Tests for database module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from soundpack.db import Database


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_creates_database_file(self, tmp_path):
        """Database creates file at specified path."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        db.close()
        assert db_path.exists()

    def test_creates_samples_table(self, tmp_path):
        """Database creates samples table with correct schema."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)

        # Verify table exists with expected columns
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(samples)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        db.close()

        expected_columns = {
            "id",
            "file_path",
            "filename",
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
            "is_oneshot",
            "source",
            "created_at",
            "updated_at",
        }
        assert expected_columns.issubset(columns)

    def test_creates_tags_table(self, tmp_path):
        """Database creates tags table with correct schema."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(tags)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        db.close()

        expected_columns = {"id", "name", "category", "created_at"}
        assert expected_columns.issubset(columns)

    def test_creates_sample_tags_table(self, tmp_path):
        """Database creates sample_tags junction table."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(sample_tags)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        db.close()

        expected_columns = {"sample_id", "tag_id", "confidence", "source"}
        assert expected_columns.issubset(columns)


class TestSampleOperations:
    """Tests for sample CRUD operations."""

    def test_add_sample_returns_id(self, tmp_path):
        """Adding a sample returns its database ID."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(
            file_path="/path/to/kick.wav",
            filename="kick.wav",
        )
        db.close()
        assert isinstance(sample_id, int)
        assert sample_id > 0

    def test_add_sample_with_all_fields(self, tmp_path):
        """Adding a sample with all audio properties."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(
            file_path="/path/to/kick.wav",
            filename="kick.wav",
            duration_ms=500,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            bpm=120.0,
            bpm_confidence=0.9,
            detected_key="C minor",
            key_confidence=0.8,
            spectral_centroid=1500.0,
            onset_strength=0.7,
            rms_energy=0.5,
            is_loop=False,
            is_oneshot=True,
            source="sample_pack_1",
        )

        sample = db.get_sample(sample_id)
        db.close()

        assert sample["file_path"] == "/path/to/kick.wav"
        assert sample["duration_ms"] == 500
        assert sample["bpm"] == 120.0
        assert sample["detected_key"] == "C minor"

    def test_get_sample_returns_none_for_missing(self, tmp_path):
        """Getting a non-existent sample returns None."""
        db = Database(tmp_path / "test.db")
        sample = db.get_sample(9999)
        db.close()
        assert sample is None

    def test_get_sample_by_path(self, tmp_path):
        """Can retrieve sample by file path."""
        db = Database(tmp_path / "test.db")
        db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        sample = db.get_sample_by_path("/path/to/kick.wav")
        db.close()
        assert sample is not None
        assert sample["filename"] == "kick.wav"

    def test_duplicate_path_raises_error(self, tmp_path):
        """Adding duplicate file path raises IntegrityError."""
        db = Database(tmp_path / "test.db")
        db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")

        with pytest.raises(sqlite3.IntegrityError):
            db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        db.close()

    def test_update_sample(self, tmp_path):
        """Can update sample fields."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")

        db.update_sample(sample_id, bpm=128.0, detected_key="D minor")

        sample = db.get_sample(sample_id)
        db.close()
        assert sample["bpm"] == 128.0
        assert sample["detected_key"] == "D minor"

    def test_remove_sample(self, tmp_path):
        """Can remove sample from database."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")

        db.remove_sample(sample_id)

        sample = db.get_sample(sample_id)
        db.close()
        assert sample is None

    def test_list_samples(self, tmp_path):
        """Can list all samples."""
        db = Database(tmp_path / "test.db")
        db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        db.add_sample(file_path="/path/to/snare.wav", filename="snare.wav")

        samples = db.list_samples()
        db.close()

        assert len(samples) == 2
        filenames = {s["filename"] for s in samples}
        assert filenames == {"kick.wav", "snare.wav"}


class TestTagOperations:
    """Tests for tag CRUD operations."""

    def test_add_tag_returns_id(self, tmp_path):
        """Adding a tag returns its ID."""
        db = Database(tmp_path / "test.db")
        tag_id = db.add_tag(name="kick", category="instrument")
        db.close()
        assert isinstance(tag_id, int)
        assert tag_id > 0

    def test_get_tag_by_name(self, tmp_path):
        """Can retrieve tag by name."""
        db = Database(tmp_path / "test.db")
        db.add_tag(name="kick", category="instrument")
        tag = db.get_tag_by_name("kick")
        db.close()
        assert tag is not None
        assert tag["category"] == "instrument"

    def test_list_tags_by_category(self, tmp_path):
        """Can list tags filtered by category."""
        db = Database(tmp_path / "test.db")
        db.add_tag(name="kick", category="instrument")
        db.add_tag(name="snare", category="instrument")
        db.add_tag(name="dark", category="mood")

        instrument_tags = db.list_tags(category="instrument")
        db.close()

        assert len(instrument_tags) == 2
        names = {t["name"] for t in instrument_tags}
        assert names == {"kick", "snare"}

    def test_duplicate_tag_name_raises_error(self, tmp_path):
        """Adding duplicate tag name raises IntegrityError."""
        db = Database(tmp_path / "test.db")
        db.add_tag(name="kick", category="instrument")

        with pytest.raises(sqlite3.IntegrityError):
            db.add_tag(name="kick", category="instrument")
        db.close()


class TestSampleTagAssociations:
    """Tests for sample-tag relationships."""

    def test_tag_sample(self, tmp_path):
        """Can associate tag with sample."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        tag_id = db.add_tag(name="kick", category="instrument")

        db.tag_sample(sample_id, tag_id, confidence=0.9, source="ai")

        tags = db.get_sample_tags(sample_id)
        db.close()

        assert len(tags) == 1
        assert tags[0]["name"] == "kick"
        assert tags[0]["confidence"] == 0.9
        assert tags[0]["source"] == "ai"

    def test_untag_sample(self, tmp_path):
        """Can remove tag from sample."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        tag_id = db.add_tag(name="kick", category="instrument")
        db.tag_sample(sample_id, tag_id)

        db.untag_sample(sample_id, tag_id)

        tags = db.get_sample_tags(sample_id)
        db.close()
        assert len(tags) == 0

    def test_get_samples_by_tag(self, tmp_path):
        """Can find samples by tag."""
        db = Database(tmp_path / "test.db")
        kick1_id = db.add_sample(file_path="/path/to/kick1.wav", filename="kick1.wav")
        kick2_id = db.add_sample(file_path="/path/to/kick2.wav", filename="kick2.wav")
        snare_id = db.add_sample(file_path="/path/to/snare.wav", filename="snare.wav")

        kick_tag_id = db.add_tag(name="kick", category="instrument")
        snare_tag_id = db.add_tag(name="snare", category="instrument")

        db.tag_sample(kick1_id, kick_tag_id)
        db.tag_sample(kick2_id, kick_tag_id)
        db.tag_sample(snare_id, snare_tag_id)

        kick_samples = db.get_samples_by_tag("kick")
        db.close()

        assert len(kick_samples) == 2
        filenames = {s["filename"] for s in kick_samples}
        assert filenames == {"kick1.wav", "kick2.wav"}

    def test_removing_sample_cascades_to_tags(self, tmp_path):
        """Removing sample also removes its tag associations."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        tag_id = db.add_tag(name="kick", category="instrument")
        db.tag_sample(sample_id, tag_id)

        db.remove_sample(sample_id)

        # Verify no orphaned associations
        conn = sqlite3.connect(tmp_path / "test.db")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sample_tags WHERE sample_id = ?", (sample_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        db.close()

        assert count == 0

    def test_has_ai_tags_returns_true_for_ai_tagged(self, tmp_path):
        """has_ai_tags returns True when sample has AI-generated tags."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        tag_id = db.add_tag(name="kick", category="instrument")
        db.tag_sample(sample_id, tag_id, source="ai")

        result = db.has_ai_tags(sample_id)
        db.close()

        assert result is True

    def test_has_ai_tags_returns_false_for_filename_tagged(self, tmp_path):
        """has_ai_tags returns False when sample only has filename-based tags."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")
        tag_id = db.add_tag(name="kick", category="instrument")
        db.tag_sample(sample_id, tag_id, source="filename")

        result = db.has_ai_tags(sample_id)
        db.close()

        assert result is False

    def test_has_ai_tags_returns_false_for_untagged(self, tmp_path):
        """has_ai_tags returns False when sample has no tags."""
        db = Database(tmp_path / "test.db")
        sample_id = db.add_sample(file_path="/path/to/kick.wav", filename="kick.wav")

        result = db.has_ai_tags(sample_id)
        db.close()

        assert result is False


class TestSampleFiltering:
    """Tests for sample filtering and search."""

    def test_filter_by_bpm_range(self, tmp_path):
        """Can filter samples by BPM range."""
        db = Database(tmp_path / "test.db")
        db.add_sample(
            file_path="/path/to/slow.wav", filename="slow.wav", bpm=80.0
        )
        db.add_sample(
            file_path="/path/to/mid.wav", filename="mid.wav", bpm=120.0
        )
        db.add_sample(
            file_path="/path/to/fast.wav", filename="fast.wav", bpm=160.0
        )

        samples = db.list_samples(bpm_min=100, bpm_max=140)
        db.close()

        assert len(samples) == 1
        assert samples[0]["filename"] == "mid.wav"

    def test_filter_by_key(self, tmp_path):
        """Can filter samples by detected key."""
        db = Database(tmp_path / "test.db")
        db.add_sample(
            file_path="/path/to/c.wav", filename="c.wav", detected_key="C minor"
        )
        db.add_sample(
            file_path="/path/to/d.wav", filename="d.wav", detected_key="D major"
        )

        samples = db.list_samples(key="C minor")
        db.close()

        assert len(samples) == 1
        assert samples[0]["filename"] == "c.wav"

    def test_filter_untagged_samples(self, tmp_path):
        """Can filter for samples without tags."""
        db = Database(tmp_path / "test.db")
        tagged_id = db.add_sample(file_path="/path/to/tagged.wav", filename="tagged.wav")
        db.add_sample(file_path="/path/to/untagged.wav", filename="untagged.wav")

        tag_id = db.add_tag(name="kick", category="instrument")
        db.tag_sample(tagged_id, tag_id)

        samples = db.list_samples(untagged=True)
        db.close()

        assert len(samples) == 1
        assert samples[0]["filename"] == "untagged.wav"

    def test_filter_by_multiple_tags(self, tmp_path):
        """Can filter samples matching multiple tags (AND logic)."""
        db = Database(tmp_path / "test.db")
        sample1_id = db.add_sample(file_path="/path/to/s1.wav", filename="s1.wav")
        sample2_id = db.add_sample(file_path="/path/to/s2.wav", filename="s2.wav")

        kick_id = db.add_tag(name="kick", category="instrument")
        dark_id = db.add_tag(name="dark", category="mood")

        db.tag_sample(sample1_id, kick_id)
        db.tag_sample(sample1_id, dark_id)
        db.tag_sample(sample2_id, kick_id)  # Only has kick, not dark

        samples = db.list_samples(tags=["kick", "dark"])
        db.close()

        assert len(samples) == 1
        assert samples[0]["filename"] == "s1.wav"
