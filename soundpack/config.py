"""Configuration management for soundpack."""

import os
from pathlib import Path
from typing import Any

import toml


def get_default_config_path() -> Path:
    """Get the default configuration file path.

    Returns:
        Path to config.toml in user's config directory.
    """
    if os.name == "nt":  # Windows
        config_dir = Path(os.environ.get("APPDATA", "~"))
    else:  # macOS/Linux
        config_dir = Path.home() / ".config"

    return config_dir / "soundpack" / "config.toml"


def get_default_db_path() -> Path:
    """Get the default database file path.

    Returns:
        Path to library.db in user's data directory.
    """
    if os.name == "nt":  # Windows
        data_dir = Path(os.environ.get("LOCALAPPDATA", "~"))
    else:  # macOS/Linux
        data_dir = Path.home() / ".local" / "share"

    return data_dir / "soundpack" / "library.db"


# Default configuration values
DEFAULT_CONFIG = {
    "api": {
        "anthropic_api_key": "",
    },
    "library": {
        "database_path": str(get_default_db_path()),
        "watch_directories": [],
    },
    "analysis": {
        "auto_analyze_on_import": True,
        "detect_loops": True,
        "min_loop_duration_ms": 1000,
    },
    "export": {
        "default_output_dir": str(Path.home() / "Music" / "Polyend" / "Packs"),
        "max_pack_size": 128,
        "filename_max_length": 16,
    },
    "tagging": {
        "auto_tag_on_import": False,
        "min_confidence_threshold": 0.5,
    },
}


class Config:
    """Configuration manager for soundpack."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file. Uses default if None.
        """
        self.config_path = Path(config_path) if config_path else get_default_config_path()
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file or return defaults.

        Returns:
            Configuration dictionary.
        """
        config = _deep_copy_dict(DEFAULT_CONFIG)

        if self.config_path.exists():
            try:
                file_config = toml.load(self.config_path)
                _deep_merge(config, file_config)
            except Exception:
                pass  # Use defaults on parse error

        return config

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            section: Config section name.
            key: Key within section.
            default: Default value if not found.

        Returns:
            Configuration value or default.
        """
        try:
            return self._config[section][key]
        except KeyError:
            return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            section: Config section name.
            key: Key within section.
            value: Value to set.
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            toml.dump(self._config, f)

    def all(self) -> dict[str, Any]:
        """Get all configuration values.

        Returns:
            Copy of configuration dictionary.
        """
        return _deep_copy_dict(self._config)

    # Convenience properties

    @property
    def api_key(self) -> str:
        """Get Anthropic API key."""
        return self.get("api", "anthropic_api_key", "")

    @property
    def db_path(self) -> Path:
        """Get database file path."""
        path = self.get("library", "database_path", str(get_default_db_path()))
        return Path(path).expanduser()

    @property
    def output_dir(self) -> Path:
        """Get default output directory for packs."""
        path = self.get(
            "export", "default_output_dir", str(Path.home() / "Music" / "Polyend" / "Packs")
        )
        return Path(path).expanduser()

    @property
    def max_pack_size(self) -> int:
        """Get maximum pack size."""
        return self.get("export", "max_pack_size", 128)


def _deep_copy_dict(d: dict) -> dict:
    """Create a deep copy of a dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override dict into base dict, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
