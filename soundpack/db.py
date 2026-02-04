"""SQLite database for sample library."""

import sqlite3
from pathlib import Path
from typing import Any


class Database:
    """SQLite database for managing sample library."""

    def __init__(self, db_path: str | Path):
        """Initialize database connection and create schema.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.executescript(
            """
            -- Core sample library
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,

                -- Audio properties (from file)
                duration_ms INTEGER,
                sample_rate INTEGER,
                channels INTEGER,
                bit_depth INTEGER,

                -- Analyzed features (from librosa)
                bpm REAL,
                bpm_confidence REAL,
                detected_key TEXT,
                key_confidence REAL,
                spectral_centroid REAL,
                onset_strength REAL,
                rms_energy REAL,

                -- Classification
                is_loop BOOLEAN DEFAULT FALSE,
                is_oneshot BOOLEAN DEFAULT TRUE,

                -- Metadata
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Tag definitions
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Sample-tag associations
            CREATE TABLE IF NOT EXISTS sample_tags (
                sample_id INTEGER REFERENCES samples(id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                PRIMARY KEY (sample_id, tag_id)
            );

            -- Generated packs history
            CREATE TABLE IF NOT EXISTS packs (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT,
                sample_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pack_samples (
                pack_id INTEGER REFERENCES packs(id) ON DELETE CASCADE,
                sample_id INTEGER REFERENCES samples(id),
                exported_filename TEXT,
                PRIMARY KEY (pack_id, sample_id)
            );

            -- User tagging preferences (for AI learning)
            CREATE TABLE IF NOT EXISTS tag_preferences (
                id INTEGER PRIMARY KEY,
                audio_feature TEXT,
                suggested_tag TEXT,
                accepted BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_samples_bpm ON samples(bpm);
            CREATE INDEX IF NOT EXISTS idx_samples_key ON samples(detected_key);
            CREATE INDEX IF NOT EXISTS idx_sample_tags_sample ON sample_tags(sample_id);
            CREATE INDEX IF NOT EXISTS idx_sample_tags_tag ON sample_tags(tag_id);
            CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
            """
        )
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    # Sample operations

    def add_sample(
        self,
        file_path: str,
        filename: str,
        duration_ms: int | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        bit_depth: int | None = None,
        bpm: float | None = None,
        bpm_confidence: float | None = None,
        detected_key: str | None = None,
        key_confidence: float | None = None,
        spectral_centroid: float | None = None,
        onset_strength: float | None = None,
        rms_energy: float | None = None,
        is_loop: bool = False,
        is_oneshot: bool = True,
        source: str | None = None,
    ) -> int:
        """Add a sample to the library.

        Args:
            file_path: Absolute path to the audio file.
            filename: Base filename.
            Other args: Optional audio properties and analysis data.

        Returns:
            The database ID of the new sample.

        Raises:
            sqlite3.IntegrityError: If file_path already exists.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO samples (
                file_path, filename, duration_ms, sample_rate, channels, bit_depth,
                bpm, bpm_confidence, detected_key, key_confidence,
                spectral_centroid, onset_strength, rms_energy,
                is_loop, is_oneshot, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                filename,
                duration_ms,
                sample_rate,
                channels,
                bit_depth,
                bpm,
                bpm_confidence,
                detected_key,
                key_confidence,
                spectral_centroid,
                onset_strength,
                rms_energy,
                is_loop,
                is_oneshot,
                source,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_sample(self, sample_id: int) -> dict[str, Any] | None:
        """Get a sample by ID.

        Args:
            sample_id: Database ID of the sample.

        Returns:
            Sample data as dict, or None if not found.
        """
        cursor = self.conn.execute(
            "SELECT * FROM samples WHERE id = ?", (sample_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_sample_by_path(self, file_path: str) -> dict[str, Any] | None:
        """Get a sample by file path.

        Args:
            file_path: Absolute path to the audio file.

        Returns:
            Sample data as dict, or None if not found.
        """
        cursor = self.conn.execute(
            "SELECT * FROM samples WHERE file_path = ?", (file_path,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_sample(self, sample_id: int, **kwargs) -> None:
        """Update sample fields.

        Args:
            sample_id: Database ID of the sample.
            **kwargs: Fields to update.
        """
        if not kwargs:
            return

        set_clause = ", ".join(f"{key} = ?" for key in kwargs)
        values = list(kwargs.values()) + [sample_id]

        self.conn.execute(
            f"UPDATE samples SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        self.conn.commit()

    def remove_sample(self, sample_id: int) -> None:
        """Remove a sample from the library.

        Args:
            sample_id: Database ID of the sample.
        """
        self.conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        self.conn.commit()

    def list_samples(
        self,
        bpm_min: float | None = None,
        bpm_max: float | None = None,
        key: str | None = None,
        untagged: bool = False,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List samples with optional filtering.

        Args:
            bpm_min: Minimum BPM (inclusive).
            bpm_max: Maximum BPM (inclusive).
            key: Filter by detected key.
            untagged: If True, only return samples without tags.
            tags: Filter by tag names (AND logic - must have all).

        Returns:
            List of sample dicts.
        """
        query = "SELECT DISTINCT s.* FROM samples s"
        conditions = []
        params: list[Any] = []

        if tags:
            # Join with sample_tags and tags for each required tag
            for i, tag_name in enumerate(tags):
                query += f"""
                    INNER JOIN sample_tags st{i} ON s.id = st{i}.sample_id
                    INNER JOIN tags t{i} ON st{i}.tag_id = t{i}.id AND t{i}.name = ?
                """
                params.append(tag_name)

        if bpm_min is not None:
            conditions.append("s.bpm >= ?")
            params.append(bpm_min)

        if bpm_max is not None:
            conditions.append("s.bpm <= ?")
            params.append(bpm_max)

        if key is not None:
            conditions.append("s.detected_key = ?")
            params.append(key)

        if untagged:
            conditions.append(
                "s.id NOT IN (SELECT DISTINCT sample_id FROM sample_tags)"
            )

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # Tag operations

    def add_tag(self, name: str, category: str) -> int:
        """Add a tag to the vocabulary.

        Args:
            name: Tag name.
            category: Tag category (instrument, character, mood, genre).

        Returns:
            The database ID of the new tag.

        Raises:
            sqlite3.IntegrityError: If tag name already exists.
        """
        cursor = self.conn.execute(
            "INSERT INTO tags (name, category) VALUES (?, ?)",
            (name, category),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_tag_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a tag by name.

        Args:
            name: Tag name.

        Returns:
            Tag data as dict, or None if not found.
        """
        cursor = self.conn.execute("SELECT * FROM tags WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_tags(self, category: str | None = None) -> list[dict[str, Any]]:
        """List tags with optional category filter.

        Args:
            category: Optional category to filter by.

        Returns:
            List of tag dicts.
        """
        if category:
            cursor = self.conn.execute(
                "SELECT * FROM tags WHERE category = ?", (category,)
            )
        else:
            cursor = self.conn.execute("SELECT * FROM tags")
        return [dict(row) for row in cursor.fetchall()]

    # Sample-tag associations

    def tag_sample(
        self,
        sample_id: int,
        tag_id: int,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> None:
        """Associate a tag with a sample.

        Args:
            sample_id: Database ID of the sample.
            tag_id: Database ID of the tag.
            confidence: Confidence score (0-1).
            source: Tag source ('manual', 'ai', 'filename').
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sample_tags (sample_id, tag_id, confidence, source)
            VALUES (?, ?, ?, ?)
            """,
            (sample_id, tag_id, confidence, source),
        )
        self.conn.commit()

    def untag_sample(self, sample_id: int, tag_id: int) -> None:
        """Remove a tag from a sample.

        Args:
            sample_id: Database ID of the sample.
            tag_id: Database ID of the tag.
        """
        self.conn.execute(
            "DELETE FROM sample_tags WHERE sample_id = ? AND tag_id = ?",
            (sample_id, tag_id),
        )
        self.conn.commit()

    def get_sample_tags(self, sample_id: int) -> list[dict[str, Any]]:
        """Get all tags for a sample.

        Args:
            sample_id: Database ID of the sample.

        Returns:
            List of tag dicts with confidence and source.
        """
        cursor = self.conn.execute(
            """
            SELECT t.*, st.confidence, st.source
            FROM tags t
            INNER JOIN sample_tags st ON t.id = st.tag_id
            WHERE st.sample_id = ?
            """,
            (sample_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def has_ai_tags(self, sample_id: int) -> bool:
        """Check if a sample has any AI-generated tags.

        Args:
            sample_id: Database ID of the sample.

        Returns:
            True if sample has at least one tag with source='ai'.
        """
        cursor = self.conn.execute(
            "SELECT 1 FROM sample_tags WHERE sample_id = ? AND source = 'ai' LIMIT 1",
            (sample_id,),
        )
        return cursor.fetchone() is not None

    def get_samples_by_tag(self, tag_name: str) -> list[dict[str, Any]]:
        """Get all samples with a specific tag.

        Args:
            tag_name: Name of the tag.

        Returns:
            List of sample dicts.
        """
        cursor = self.conn.execute(
            """
            SELECT s.*
            FROM samples s
            INNER JOIN sample_tags st ON s.id = st.sample_id
            INNER JOIN tags t ON st.tag_id = t.id
            WHERE t.name = ?
            """,
            (tag_name,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # Stats operations

    def get_stats(self) -> dict[str, Any]:
        """Get library statistics.

        Returns:
            Dict with various statistics about the library.
        """
        stats: dict[str, Any] = {}

        # Total samples
        cursor = self.conn.execute("SELECT COUNT(*) FROM samples")
        stats["total_samples"] = cursor.fetchone()[0]

        # Analyzed samples (have BPM data)
        cursor = self.conn.execute("SELECT COUNT(*) FROM samples WHERE bpm IS NOT NULL")
        stats["analyzed_samples"] = cursor.fetchone()[0]

        # Samples with tags
        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT sample_id) FROM sample_tags"
        )
        stats["tagged_samples"] = cursor.fetchone()[0]

        # AI-tagged samples (source = 'ai')
        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT sample_id) FROM sample_tags WHERE source = 'ai'"
        )
        stats["ai_tagged_samples"] = cursor.fetchone()[0]

        # Total tags in vocabulary
        cursor = self.conn.execute("SELECT COUNT(*) FROM tags")
        stats["total_tags"] = cursor.fetchone()[0]

        # Tag counts by category
        cursor = self.conn.execute(
            """
            SELECT category, COUNT(*) as count
            FROM tags
            GROUP BY category
            ORDER BY count DESC
            """
        )
        stats["tags_by_category"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Most used tags (top 10)
        cursor = self.conn.execute(
            """
            SELECT t.name, t.category, COUNT(st.sample_id) as usage_count
            FROM tags t
            LEFT JOIN sample_tags st ON t.id = st.tag_id
            GROUP BY t.id
            ORDER BY usage_count DESC
            LIMIT 10
            """
        )
        stats["top_tags"] = [
            {"name": row[0], "category": row[1], "count": row[2]}
            for row in cursor.fetchall()
        ]

        # Total duration
        cursor = self.conn.execute("SELECT SUM(duration_ms) FROM samples")
        total_ms = cursor.fetchone()[0] or 0
        stats["total_duration_ms"] = total_ms

        # Packs created
        cursor = self.conn.execute("SELECT COUNT(*) FROM packs")
        stats["packs_created"] = cursor.fetchone()[0]

        return stats
