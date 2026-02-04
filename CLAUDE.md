# Polyend Play Plus Sound Pack Manager

A Python CLI tool for cataloging, tagging, and organizing audio samples with AI-assisted metadata generation. Generates curated sound packs for the Polyend Play Plus hardware sampler.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run CLI
soundpack --help
```

## Project Structure

```
soundpack/
├── __init__.py
├── cli.py              # Click command definitions
├── db.py               # SQLite schema and queries
├── audio.py            # librosa analysis functions
├── tagger.py           # AI-assisted tagging logic
├── generator.py        # Pack generation from prompts
├── exporter.py         # Play Plus folder output
└── config.py           # Settings management
tests/
pyproject.toml
```

## Key Constraints

### Polyend Play Plus Hardware Limits
- **Audio format**: WAV (mono/stereo, 44.1kHz, 16/24-bit)
- **Filename display**: 16 characters max (excluding .wav)
- **Samples per folder**: 255 max
- **Folder structure**: Flat (no nesting)

## Architecture

### Tech Stack
- Python 3.10+
- Click (CLI framework)
- SQLite (database)
- librosa (audio analysis)
- Anthropic Claude API (AI tagging)

### Database Tables
- `samples` - Core sample library with audio properties and analyzed features
- `tags` - Tag definitions with categories (instrument, character, mood, genre)
- `sample_tags` - Sample-tag associations with confidence scores
- `packs` - Generated pack history
- `pack_samples` - Pack-sample mappings with exported filenames
- `tag_preferences` - User tagging preferences for AI learning

## CLI Commands

```bash
# Import & Library
soundpack import <directory> [--recursive] [--analyze] [--tag]
soundpack analyze [--all] [--missing] [sample_id...]
soundpack list [--tag TAG] [--bpm MIN-MAX] [--key KEY] [--untagged]
soundpack info <sample_id_or_path>
soundpack remove <sample_id_or_path> [--delete-file]

# Tagging
soundpack tag [--batch SIZE] [--untagged-only]
soundpack tag <sample_id_or_path> <tag1> [tag2...]
soundpack tags [--category CATEGORY]
soundpack tags add <name> --category <category>

# Search
soundpack search "dark kicks with punch"
soundpack similar <sample_id_or_path> [--limit 10]

# Pack Generation
soundpack generate "80s drums with dark analog bass" --name "80s_Dark" --max 64
soundpack export --tag kick --tag 808 --name "808_Kicks"
soundpack packs
soundpack packs rebuild <pack_id> [--add "more snares"]

# Config
soundpack config set anthropic_api_key <key>
soundpack config show
```

## Implementation Phases

1. **Core Infrastructure**: CLI skeleton, SQLite setup, basic import, audio analysis
2. **Tagging System**: Manual tagging, tag vocabulary, AI-assisted tagging
3. **Pack Generation**: Search/filter, prompt parsing, pack generation, Play Plus export
4. **Polish**: Config management, error handling, docs, tests

## Tag Categories

- **instrument**: kick, snare, hihat, bass, pad, vocal, fx, loop...
- **character**: punchy, soft, distorted, analog, digital...
- **mood**: dark, bright, warm, aggressive, mellow...
- **genre**: house, techno, ambient, hiphop, trap...

## Testing

```bash
pytest tests/ -v --cov=soundpack
```

## Config Location

`~/.config/soundpack/config.toml`
