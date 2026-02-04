# Polyend Play+ Sound Pack Manager

A CLI tool for cataloging, tagging, and organizing audio samples with AI-assisted metadata generation. Generates curated sound packs optimized for the [Polyend Play+](https://polyend.com/play-plus/) hardware sampler.

## Features

- **Smart Import**: Scan directories for WAV files, extract audio properties, and analyze BPM/key
- **AI-Assisted Tagging**: Automatically tag samples using Claude AI based on filename and audio characteristics
- **Natural Language Pack Generation**: Create packs from prompts like "dark 808 kicks" or "dreamy ambient pads"
- **Play+ Optimized Export**: Organizes samples into folders (Kick, Snare, HiHat, etc.) for Beat Fill compatibility
- **Local SQLite Database**: Fast searching and filtering across your entire sample library

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/polyend-plus-manager.git
cd polyend-plus-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Import samples from a directory
soundpack import ~/Music/Samples --recursive --analyze

# Auto-tag samples using AI (requires Anthropic API key)
soundpack config set api.anthropic_api_key sk-ant-...
soundpack autotag --all --ai

# Generate a pack from a natural language prompt
soundpack generate "punchy techno kicks with dark 808 bass" --name TechnoDark

# Preview without exporting
soundpack generate "ambient pads and textures" --dry-run

# View library statistics
soundpack stats
```

## Commands

| Command | Description |
|---------|-------------|
| `import <dir>` | Import WAV files into the library |
| `list` | List samples with optional filters |
| `info <id>` | Show detailed sample info |
| `analyze` | Run BPM/key detection on samples |
| `autotag` | Auto-tag samples from filename or AI |
| `tag <id> <tags...>` | Manually tag a sample |
| `search <query>` | Search samples by tags/attributes |
| `generate <prompt>` | Generate a pack from natural language |
| `export` | Export filtered samples as a pack |
| `stats` | Show library statistics |
| `config` | Manage configuration |

## Polyend Play+ Constraints

The tool respects Play+ hardware limits:
- **Filename display**: 16 characters max (excluding .wav)
- **Samples per folder**: 255 max
- **Audio format**: WAV (mono/stereo, 44.1kHz, 16/24-bit)
- **Beat Fill**: Requires 5+ samples in Kick, Snare, HiHat folders

## Configuration

Config file location: `~/.config/soundpack/config.toml`

```toml
[api]
anthropic_api_key = "sk-ant-..."

[export]
default_output_dir = "~/Music/Polyend/Packs"
max_pack_size = 128
```

## Requirements

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/) (optional, for AI features)

## License

MIT License - see [LICENSE](LICENSE) for details.
