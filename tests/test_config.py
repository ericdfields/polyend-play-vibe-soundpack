"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from soundpack.config import Config, get_default_config_path, get_default_db_path


class TestConfigPaths:
    """Tests for config path utilities."""

    def test_default_config_path_in_user_config(self):
        """Default config path is in user config directory."""
        path = get_default_config_path()
        assert ".config" in str(path) or "AppData" in str(path)
        assert "soundpack" in str(path)
        assert path.name == "config.toml"

    def test_default_db_path_in_user_data(self):
        """Default database path is in user data directory."""
        path = get_default_db_path()
        assert ".local/share" in str(path) or "AppData" in str(path)
        assert "soundpack" in str(path)
        assert path.name == "library.db"


class TestConfigInit:
    """Tests for config initialization."""

    def test_creates_config_file_if_missing(self, tmp_path):
        """Creates config file with defaults if it doesn't exist."""
        config_path = tmp_path / "config.toml"
        config = Config(config_path)
        config.save()

        assert config_path.exists()

    def test_loads_existing_config(self, tmp_path):
        """Loads values from existing config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[api]
anthropic_api_key = "test-key-123"

[export]
max_pack_size = 200
"""
        )

        config = Config(config_path)
        assert config.get("api", "anthropic_api_key") == "test-key-123"
        assert config.get("export", "max_pack_size") == 200


class TestConfigGetSet:
    """Tests for getting and setting config values."""

    def test_get_returns_default_value(self, tmp_path):
        """Get returns default value when key doesn't exist."""
        config = Config(tmp_path / "config.toml")
        assert config.get("nonexistent", "key", default="fallback") == "fallback"

    def test_set_creates_section(self, tmp_path):
        """Set creates section if it doesn't exist."""
        config = Config(tmp_path / "config.toml")
        config.set("new_section", "new_key", "new_value")
        assert config.get("new_section", "new_key") == "new_value"

    def test_set_persists_after_save(self, tmp_path):
        """Set values persist after save and reload."""
        config_path = tmp_path / "config.toml"
        config = Config(config_path)
        config.set("api", "anthropic_api_key", "my-secret-key")
        config.save()

        # Reload
        config2 = Config(config_path)
        assert config2.get("api", "anthropic_api_key") == "my-secret-key"

    def test_get_api_key(self, tmp_path):
        """Convenience method for API key."""
        config = Config(tmp_path / "config.toml")
        config.set("api", "anthropic_api_key", "test-key")
        assert config.api_key == "test-key"

    def test_get_db_path(self, tmp_path):
        """Convenience method for database path."""
        config = Config(tmp_path / "config.toml")
        db_path = config.db_path
        assert isinstance(db_path, Path)
        assert db_path.name == "library.db"

    def test_custom_db_path(self, tmp_path):
        """Can set custom database path."""
        config = Config(tmp_path / "config.toml")
        custom_path = str(tmp_path / "custom.db")
        config.set("library", "database_path", custom_path)
        assert config.db_path == Path(custom_path)

    def test_get_output_dir(self, tmp_path):
        """Convenience method for output directory."""
        config = Config(tmp_path / "config.toml")
        output_dir = config.output_dir
        assert isinstance(output_dir, Path)

    def test_get_max_pack_size(self, tmp_path):
        """Convenience method for max pack size."""
        config = Config(tmp_path / "config.toml")
        assert config.max_pack_size == 128  # default

        config.set("export", "max_pack_size", 64)
        assert config.max_pack_size == 64


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_api_section(self, tmp_path):
        """API section has expected defaults."""
        config = Config(tmp_path / "config.toml")
        assert config.get("api", "anthropic_api_key") == ""

    def test_default_export_section(self, tmp_path):
        """Export section has expected defaults."""
        config = Config(tmp_path / "config.toml")
        assert config.get("export", "max_pack_size") == 128
        assert config.get("export", "filename_max_length") == 16

    def test_default_analysis_section(self, tmp_path):
        """Analysis section has expected defaults."""
        config = Config(tmp_path / "config.toml")
        assert config.get("analysis", "auto_analyze_on_import") is True
        assert config.get("analysis", "min_loop_duration_ms") == 1000

    def test_default_tagging_section(self, tmp_path):
        """Tagging section has expected defaults."""
        config = Config(tmp_path / "config.toml")
        assert config.get("tagging", "auto_tag_on_import") is False
        assert config.get("tagging", "min_confidence_threshold") == 0.5


class TestConfigAll:
    """Tests for listing all config values."""

    def test_all_returns_dict(self, tmp_path):
        """all() returns dictionary of all config sections."""
        config = Config(tmp_path / "config.toml")
        all_config = config.all()

        assert isinstance(all_config, dict)
        assert "api" in all_config
        assert "export" in all_config
        assert "library" in all_config
