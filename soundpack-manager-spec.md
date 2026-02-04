# Polyend Play Plus Sound Pack Manager

## Project Specification v1.0

---

## Overview

A Python CLI tool for cataloging, tagging, and organizing audio samples with AI-assisted metadata generation. The tool enables users to build a searchable library of samples and generate curated sound packs for the Polyend Play Plus hardware sampler using natural language prompts.

### Core Value Proposition

- **AI-assisted tagging** reduces the manual effort of cataloging hundreds/thousands of samples
- **Audio analysis** extracts objective characteristics (BPM, key, spectral features) automatically
- **Natural language pack generation** turns prompts like "dark 808 drums with analog bass, 90-110 BPM" into ready-to-load sample packs
- **Play Plus compatibility** handles folder structure, filename limits, and format requirements automatically

---

## Target Hardware: Polyend Play Plus

### Specifications & Constraints

| Constraint | Value | Implementation Note |
|------------|-------|---------------------|
| Audio format | WAV (mono/stereo, 44.1kHz, 16/24-bit) | Validate on import, convert if needed |
| Filename display | 16 characters | Truncate intelligently, preserve uniqueness |
| Samples per folder | 255 max | Enforce during pack generation |
| Folder structure | Flat (no nesting) | Single level of organization |
| Sample length | No hard limit | Optional user-defined max length filter |

### Pack Output Structure

```
MyPack/
├── kick_808_dark_01.wav
├── kick_808_dark_02.wav
├── snare_punchy_03.wav
├── bass_analog_04.wav
└── ... (up to 255 files)
```

---

## Technical Architecture

### Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.10+ | Rich audio analysis ecosystem |
| CLI Framework | Click | Clean decorator syntax, built-in help |
| Database | SQLite | Single file, no server, portable |
| Audio Analysis | librosa | Industry standard, BPM/key/spectral analysis |
| AI Integration | Anthropic Claude API | Intelligent tagging suggestions |
| Audio Playback | sounddevice (optional) | Preview during tagging sessions |

### Directory Structure

```
soundpack-manager/
├── soundpack/
│   ├── __init__.py
│   ├── cli.py              # Click command definitions
│   ├── db.py               # SQLite schema and queries
│   ├── audio.py            # librosa analysis functions
│   ├── tagger.py           # AI-assisted tagging logic
│   ├── generator.py        # Pack generation from prompts
│   ├── exporter.py         # Play Plus folder output
│   └── config.py           # Settings management
├── tests/
├── pyproject.toml
└── README.md
```

---

## Database Schema

### SQLite Tables

```sql
-- Core sample library
CREATE TABLE samples (
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
    detected_key TEXT,           -- e.g., "C minor", "F# major"
    key_confidence REAL,
    spectral_centroid REAL,      -- brightness indicator
    onset_strength REAL,         -- transient sharpness
    rms_energy REAL,             -- loudness
    
    -- Classification
    is_loop BOOLEAN DEFAULT FALSE,
    is_oneshot BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    source TEXT,                 -- where sample came from
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tag definitions
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,      -- 'instrument', 'character', 'mood', 'genre'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample-tag associations
CREATE TABLE sample_tags (
    sample_id INTEGER REFERENCES samples(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    confidence REAL DEFAULT 1.0, -- AI confidence or 1.0 for manual
    source TEXT DEFAULT 'manual', -- 'manual', 'ai', 'filename'
    PRIMARY KEY (sample_id, tag_id)
);

-- Generated packs history
CREATE TABLE packs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT,                 -- original generation prompt
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pack_samples (
    pack_id INTEGER REFERENCES packs(id) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES samples(id),
    exported_filename TEXT,      -- truncated name used in export
    PRIMARY KEY (pack_id, sample_id)
);

-- User tagging preferences (for AI learning)
CREATE TABLE tag_preferences (
    id INTEGER PRIMARY KEY,
    audio_feature TEXT,          -- e.g., 'low_spectral_centroid'
    suggested_tag TEXT,
    accepted BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

```sql
CREATE INDEX idx_samples_bpm ON samples(bpm);
CREATE INDEX idx_samples_key ON samples(detected_key);
CREATE INDEX idx_sample_tags_sample ON sample_tags(sample_id);
CREATE INDEX idx_sample_tags_tag ON sample_tags(tag_id);
CREATE INDEX idx_tags_category ON tags(category);
```

---

## Tag Vocabulary

### Predefined Categories

#### Instrument (category: 'instrument')
```
kick, snare, hihat, clap, rim, tom, cymbal, shaker, tambourine,
conga, bongo, cowbell, percussion,
bass, sub, lead, pad, pluck, stab, chord, arp,
vocal, vox, chant, speech,
fx, riser, downlifter, impact, texture, noise, atmosphere,
loop, break, fill
```

#### Character (category: 'character')
```
punchy, soft, hard, tight, loose,
distorted, saturated, clean, dry, wet,
acoustic, electronic, analog, digital,
long, short, sustained, staccato,
layered, thin, fat, wide, narrow
```

#### Mood (category: 'mood')
```
dark, bright, warm, cold,
aggressive, mellow, energetic, calm,
eerie, haunting, uplifting, melancholic,
dirty, clean, lo-fi, hi-fi,
vintage, modern, futuristic, retro
```

#### Genre (category: 'genre')
```
house, techno, ambient, hiphop, trap,
dnb, dubstep, breaks, electro, disco,
industrial, experimental, cinematic
```

---

## CLI Commands

### Import & Library Management

```bash
# Scan directory and add samples to library
soundpack import <directory> [--recursive] [--analyze] [--tag]
  --recursive       Scan subdirectories
  --analyze         Run audio analysis (BPM, key, spectral)
  --tag             Run AI-assisted tagging after import

# Re-analyze existing samples
soundpack analyze [--all] [--missing] [sample_id...]
  --all             Analyze all samples
  --missing         Only samples without analysis data

# List samples with filters
soundpack list [--tag TAG] [--bpm MIN-MAX] [--key KEY] [--untagged]
  --tag TAG         Filter by tag (can repeat)
  --bpm 90-120      Filter by BPM range
  --key "C minor"   Filter by detected key
  --untagged        Show only untagged samples

# Show sample details
soundpack info <sample_id_or_path>

# Remove sample from library (doesn't delete file)
soundpack remove <sample_id_or_path> [--delete-file]
```

### Tagging

```bash
# Interactive tagging session with AI suggestions
soundpack tag [--batch SIZE] [--untagged-only]
  --batch 20        Process N samples per session
  --untagged-only   Only show samples without tags

# Quick tag specific samples
soundpack tag <sample_id_or_path> <tag1> [tag2] [tag3...]

# List all tags
soundpack tags [--category CATEGORY]

# Add custom tag to vocabulary
soundpack tags add <name> --category <category>
```

### Search & Discovery

```bash
# Search by tags and features
soundpack search "dark kicks with punch"
soundpack search --tag kick --tag dark --bpm 90-110

# Find similar samples
soundpack similar <sample_id_or_path> [--limit 10]
```

### Pack Generation

```bash
# Generate pack from natural language prompt
soundpack generate "80s drum machine sounds with dark analog bass"
  --name "80s_Dark"           Pack name (auto-generated if omitted)
  --max 64                    Maximum samples (default: 64)
  --output <directory>        Output location (default: ./packs/)
  --dry-run                   Show what would be included without exporting

# Export existing search as pack
soundpack export --tag kick --tag 808 --name "808_Kicks" --output ./packs/

# List generated packs
soundpack packs

# Regenerate/modify existing pack
soundpack packs rebuild <pack_id> [--add "more snares"] [--remove "claps"]
```

### Configuration

```bash
# Configure API key and settings
soundpack config set anthropic_api_key <key>
soundpack config set default_output_dir ~/polyend/packs
soundpack config set max_pack_size 128
soundpack config show
```

---

## Audio Analysis Pipeline

### librosa Feature Extraction

```python
def analyze_sample(file_path: str) -> dict:
    """
    Extract audio features using librosa.
    
    Returns:
        dict with keys:
        - duration_ms: int
        - sample_rate: int
        - bpm: float
        - bpm_confidence: float (0-1)
        - detected_key: str (e.g., "C minor")
        - key_confidence: float (0-1)
        - spectral_centroid: float (Hz, indicates brightness)
        - onset_strength: float (transient sharpness)
        - rms_energy: float (perceived loudness)
        - is_loop: bool (based on length + repetition detection)
    """
```

### Feature-to-Tag Mapping

The AI tagger uses audio features as context:

| Feature | Low Value Suggests | High Value Suggests |
|---------|-------------------|---------------------|
| spectral_centroid | dark, warm, bass, sub | bright, airy, hi-freq |
| onset_strength | soft, pad, sustained | punchy, percussive, transient |
| rms_energy | quiet, subtle, soft | loud, aggressive, powerful |
| duration | one-shot, hit, stab | loop, sustained, pad |

---

## AI-Assisted Tagging

### Prompt Strategy

When analyzing a sample, send to Claude:

```
You are helping categorize audio samples for a music production library.

Sample information:
- Filename: {filename}
- Duration: {duration_ms}ms
- BPM: {bpm} (confidence: {bpm_confidence})
- Detected key: {detected_key}
- Spectral centroid: {spectral_centroid} Hz (higher = brighter)
- Onset strength: {onset_strength} (higher = more percussive)
- RMS energy: {rms_energy} (higher = louder)

Available tags by category:
- Instrument: {instrument_tags}
- Character: {character_tags}  
- Mood: {mood_tags}
- Genre: {genre_tags}

Based on the filename and audio characteristics, suggest appropriate tags.
Return JSON: {"instrument": [...], "character": [...], "mood": [...], "genre": [...]}

Only suggest tags you're confident about. It's better to suggest fewer accurate tags than many uncertain ones.
```

### Confidence Scoring

- **Filename-derived tags**: confidence 0.9 (filenames are usually accurate)
- **AI-suggested tags**: confidence 0.6-0.8 based on feature clarity
- **User-confirmed tags**: confidence 1.0

### Learning from Corrections

Store user corrections in `tag_preferences` table:

```python
# When user rejects AI suggestion
INSERT INTO tag_preferences (audio_feature, suggested_tag, accepted)
VALUES ('low_spectral_centroid', 'dark', FALSE);

# Use accumulated preferences to refine future prompts
# "Note: User has previously rejected 'dark' for low spectral centroid samples"
```

---

## Pack Generation Logic

### Prompt Parsing

The generator interprets natural language prompts:

```python
def parse_generation_prompt(prompt: str) -> dict:
    """
    Use Claude to parse prompt into structured query.
    
    Input: "80s drum machine sounds with dark analog bass, 90-110 BPM"
    
    Output: {
        "required_tags": ["808", "drum machine", "analog", "bass"],
        "mood_tags": ["dark"],
        "bpm_range": [90, 110],
        "key": None,
        "sample_types": ["kick", "snare", "hihat", "clap", "bass"],
        "exclude_tags": [],
        "balance": {
            "drums": 0.6,
            "bass": 0.3,
            "other": 0.1
        }
    }
    """
```

### Selection Algorithm

1. **Query matching samples** based on parsed criteria
2. **Score samples** by relevance (tag match count, feature alignment)
3. **Balance selection** to avoid over-representation (e.g., not all kicks)
4. **Enforce limits** (max samples, Play Plus constraints)
5. **Generate unique filenames** (truncate to 16 chars, ensure uniqueness)

### Filename Truncation Strategy

```python
def generate_export_filename(original: str, existing: set) -> str:
    """
    Create Play Plus compatible filename.
    
    Rules:
    - Max 16 characters (excluding .wav)
    - Preserve key identifiers (instrument type, number)
    - Ensure uniqueness within pack
    
    Examples:
    - "808_kick_heavy_distorted_01.wav" -> "808_kck_hvy_01.wav"
    - "ambient_pad_lush_warm_long.wav" -> "amb_pad_lush.wav"
    """
```

---

## Configuration

### Config File Location

```
~/.config/soundpack/config.toml
```

### Default Configuration

```toml
[api]
anthropic_api_key = ""  # Required for AI features

[library]
database_path = "~/.local/share/soundpack/library.db"
watch_directories = []  # Auto-import from these paths

[analysis]
auto_analyze_on_import = true
detect_loops = true
min_loop_duration_ms = 1000

[export]
default_output_dir = "~/Music/Polyend/Packs"
max_pack_size = 128
filename_max_length = 16

[tagging]
auto_tag_on_import = false  # Requires API key
min_confidence_threshold = 0.5
```

---

## Error Handling

### Common Scenarios

| Scenario | Behavior |
|----------|----------|
| Invalid audio file | Log warning, skip file, continue import |
| API key missing | Disable AI features, prompt user to configure |
| API rate limit | Exponential backoff, queue remaining samples |
| Duplicate file path | Update existing record, log info |
| Pack exceeds 255 samples | Warn user, offer to split or truncate |
| Unsupported format | Offer to convert (requires ffmpeg) |

---

## Future Enhancements (Out of Scope for v1)

- **Waveform visualization** in terminal (using rich or similar)
- **Audio preview** during tagging (sounddevice integration)
- **Web UI** for visual library browsing
- **Sync with Polyend** device detection and direct transfer
- **Sample similarity search** using audio embeddings
- **Batch format conversion** (mp3/aiff -> wav)
- **Smart playlists** that auto-update based on criteria

---

## Development Setup

### Prerequisites

```bash
# System dependencies (macOS)
brew install libsndfile ffmpeg

# System dependencies (Ubuntu/Debian)
apt-get install libsndfile1 ffmpeg

# Python environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Dependencies (pyproject.toml)

```toml
[project]
name = "soundpack-manager"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "anthropic>=0.18",
    "librosa>=0.10",
    "soundfile>=0.12",
    "numpy>=1.24",
    "rich>=13.0",        # Pretty terminal output
    "toml>=0.10",        # Config file parsing
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "ruff",
]
audio = [
    "sounddevice>=0.4",  # Optional audio playback
]

[project.scripts]
soundpack = "soundpack.cli:main"
```

### Testing

```bash
pytest tests/ -v --cov=soundpack
```

---

## Implementation Priority

### Phase 1: Core Infrastructure
1. CLI skeleton with Click
2. SQLite database setup
3. Basic import (scan directory, add to DB)
4. Audio analysis with librosa

### Phase 2: Tagging System
5. Manual tagging commands
6. Tag vocabulary management
7. AI-assisted tagging integration
8. Interactive tagging session

### Phase 3: Pack Generation
9. Search/filter commands
10. Prompt parsing with Claude
11. Pack generation logic
12. Play Plus export (folder structure, filename truncation)

### Phase 4: Polish
13. Configuration management
14. Error handling refinement
15. Documentation
16. Testing coverage

---

## Appendix: librosa Code Examples

### BPM Detection

```python
import librosa

def detect_bpm(file_path: str) -> tuple[float, float]:
    y, sr = librosa.load(file_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Confidence based on beat strength consistency
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    confidence = min(1.0, onset_env.std() / onset_env.mean())
    
    return float(tempo), confidence
```

### Key Detection

```python
def detect_key(file_path: str) -> tuple[str, float]:
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Krumhansl-Schmuckler key-finding algorithm
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Simplified: find dominant pitch class
    pitch_class = chroma.mean(axis=1)
    key_idx = pitch_class.argmax()
    
    # Major/minor detection (simplified)
    # Full implementation would use key profiles
    
    return f"{key_names[key_idx]} major", float(pitch_class.max())
```

### Spectral Features

```python
def extract_spectral_features(file_path: str) -> dict:
    y, sr = librosa.load(file_path)
    
    return {
        'spectral_centroid': float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        'onset_strength': float(librosa.onset.onset_strength(y=y, sr=sr).mean()),
        'rms_energy': float(librosa.feature.rms(y=y).mean()),
    }
```
