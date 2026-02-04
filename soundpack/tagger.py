"""AI-assisted tagging using Claude API."""

import json
import re
from typing import Any

import anthropic

# Use Sonnet for high-quality tagging (Haiku was too shallow)
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Tag vocabulary by category
TAG_VOCABULARY = {
    "instrument": {
        "kick",
        "snare",
        "hihat",
        "clap",
        "rim",
        "tom",
        "cymbal",
        "shaker",
        "tambourine",
        "conga",
        "bongo",
        "cowbell",
        "percussion",
        "bass",
        "sub",
        "lead",
        "pad",
        "pluck",
        "stab",
        "chord",
        "arp",
        "vocal",
        "vox",
        "chant",
        "speech",
        "fx",
        "riser",
        "downlifter",
        "impact",
        "texture",
        "noise",
        "atmosphere",
        "loop",
        "break",
        "fill",
        "808",
        "909",
    },
    "character": {
        "punchy",
        "soft",
        "hard",
        "tight",
        "loose",
        "distorted",
        "saturated",
        "clean",
        "dry",
        "wet",
        "acoustic",
        "electronic",
        "analog",
        "digital",
        "long",
        "short",
        "sustained",
        "staccato",
        "layered",
        "thin",
        "fat",
        "wide",
        "narrow",
    },
    "mood": {
        "dark",
        "bright",
        "warm",
        "cold",
        "aggressive",
        "mellow",
        "energetic",
        "calm",
        "eerie",
        "haunting",
        "uplifting",
        "melancholic",
        "dirty",
        "lo-fi",
        "hi-fi",
        "vintage",
        "modern",
        "futuristic",
        "retro",
    },
    "genre": {
        "house",
        "techno",
        "ambient",
        "hiphop",
        "trap",
        "dnb",
        "dubstep",
        "breaks",
        "electro",
        "disco",
        "industrial",
        "experimental",
        "cinematic",
    },
}

# Flatten vocabulary for quick lookup
ALL_VALID_TAGS = set()
for tags in TAG_VOCABULARY.values():
    ALL_VALID_TAGS.update(tags)


def get_anthropic_client(api_key: str | None = None) -> anthropic.Anthropic | None:
    """Get Anthropic client if API key is available.

    Args:
        api_key: API key or None to use environment variable.

    Returns:
        Anthropic client or None if no key available.
    """
    try:
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception:
        return None


def extract_tags_from_filename(filename: str) -> list[str]:
    """Extract tags from filename using vocabulary matching.

    Args:
        filename: The filename to parse.

    Returns:
        List of matched tags.
    """
    # Remove extension and split by common separators
    name = filename.rsplit(".", 1)[0].lower()
    words = re.split(r"[_\-\s]+", name)

    tags = []
    for word in words:
        # Skip numbers
        if word.isdigit() or re.match(r"^v?\d+$", word):
            continue

        if word in ALL_VALID_TAGS:
            tags.append(word)

    return tags


def build_tagging_prompt(sample_info: dict[str, Any]) -> str:
    """Build prompt for Claude to suggest tags.

    Args:
        sample_info: Dict with sample metadata and analysis.

    Returns:
        Formatted prompt string.
    """
    # Format audio features
    features = []
    if sample_info.get("duration_ms"):
        features.append(f"Duration: {sample_info['duration_ms']}ms")
    if sample_info.get("bpm"):
        conf = sample_info.get("bpm_confidence", 0)
        features.append(f"BPM: {sample_info['bpm']:.1f} (confidence: {conf:.2f})")
    if sample_info.get("detected_key"):
        conf = sample_info.get("key_confidence", 0)
        features.append(f"Key: {sample_info['detected_key']} (confidence: {conf:.2f})")
    if sample_info.get("spectral_centroid"):
        features.append(f"Spectral centroid: {sample_info['spectral_centroid']:.1f} Hz")
    if sample_info.get("onset_strength"):
        features.append(f"Onset strength: {sample_info['onset_strength']:.3f}")
    if sample_info.get("rms_energy"):
        features.append(f"RMS energy: {sample_info['rms_energy']:.3f}")

    features_str = "\n".join(f"- {f}" for f in features) if features else "No analysis data available"

    # Format tag vocabulary
    vocab_str = ""
    for category, tags in TAG_VOCABULARY.items():
        vocab_str += f"- {category.capitalize()}: {', '.join(sorted(tags))}\n"

    prompt = f"""You are helping categorize audio samples for a music production library.

Sample information:
- Filename: {sample_info.get('filename', 'unknown')}
{features_str}

Available tags by category:
{vocab_str}
Based on the filename and audio characteristics, suggest appropriate tags.
Return ONLY a JSON object in this exact format:
{{"instrument": [...], "character": [...], "mood": [...], "genre": [...]}}

Only suggest tags from the vocabulary above. It's better to suggest fewer accurate tags than many uncertain ones.
For spectral centroid: higher values indicate brighter sounds, lower values indicate darker/bassier sounds.
For onset strength: higher values indicate more percussive/transient sounds.
"""
    return prompt


def parse_ai_response(response: str) -> list[str]:
    """Parse AI response to extract suggested tags.

    Args:
        response: Raw response from Claude.

    Returns:
        List of valid tag names.
    """
    if not response:
        return []

    # Try to find JSON in the response
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    # Collect all tags from all categories
    tags = []
    for category in ["instrument", "character", "mood", "genre"]:
        category_tags = data.get(category, [])
        if isinstance(category_tags, list):
            tags.extend(category_tags)

    # Filter to only valid tags
    return [tag.lower() for tag in tags if tag.lower() in ALL_VALID_TAGS]


def get_filename_pattern(filename: str) -> str:
    """Extract a pattern from filename by removing trailing numbers.

    This groups files like kick_01.wav, kick_02.wav into pattern "kick".

    Args:
        filename: The filename to extract pattern from.

    Returns:
        Pattern string (filename with numbers and extension stripped).
    """
    # Remove extension
    name = filename.rsplit(".", 1)[0].lower()
    # Remove trailing numbers and separators (kick_01 -> kick, snare-15 -> snare)
    pattern = re.sub(r"[-_\s]*\d+$", "", name)
    # Also handle patterns like "01_kick" -> "kick"
    pattern = re.sub(r"^\d+[-_\s]*", "", pattern)
    return pattern


def suggest_tags(
    sample_info: dict[str, Any],
    api_key: str | None = None,
    use_ai: bool = True,
    pattern_cache: dict[str, list[str]] | None = None,
) -> list[str]:
    """Suggest tags for a sample using filename analysis and optionally AI.

    Args:
        sample_info: Sample metadata dict.
        api_key: Anthropic API key (optional).
        use_ai: Whether to use AI for additional suggestions.
        pattern_cache: Optional dict to cache AI results by filename pattern.
            Pass a shared dict across calls to enable caching.

    Returns:
        List of suggested tag names.
    """
    tags = set()

    # Always extract from filename
    filename = sample_info.get("filename", "")
    filename_tags = extract_tags_from_filename(filename)
    tags.update(filename_tags)

    # Use AI if requested and available
    if use_ai:
        # Check pattern cache first
        pattern = get_filename_pattern(filename)
        if pattern_cache is not None and pattern in pattern_cache:
            tags.update(pattern_cache[pattern])
            return list(tags)

        client = get_anthropic_client(api_key)
        if client:
            try:
                prompt = build_tagging_prompt(sample_info)
                response = client.messages.create(
                    model=DEFAULT_MODEL,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

                if response.content and len(response.content) > 0:
                    ai_tags = parse_ai_response(response.content[0].text)
                    tags.update(ai_tags)

                    # Cache the AI result for this pattern
                    if pattern_cache is not None and pattern:
                        pattern_cache[pattern] = ai_tags
            except Exception:
                pass  # Fall back to filename-only tags

    return list(tags)
