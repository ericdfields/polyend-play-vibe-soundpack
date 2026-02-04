"""Pack generation from natural language prompts."""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from soundpack.tagger import DEFAULT_MODEL
from soundpack.exporter import (
    get_folder_for_tags,
    PERCUSSION_FOLDERS,
    INSTRUMENT_FOLDERS,
    MIN_SAMPLES_PER_PERC_FOLDER,
)

# Folder allocation weights for pack generation
# Higher weight = more samples allocated to that folder
# Kick, Snare, HiHat are highest priority for Beat Fill
# Synth, Vocal, Bass are medium priority
# FX, Perc fill remaining space with lowest priority
FOLDER_WEIGHTS = {
    "Kick": 5,
    "Snare": 5,
    "HiHat": 5,
    "Synth": 3,
    "Vocal": 3,
    "Bass": 3,
    "Perc": 2,
    "FX": 1,
}

# Beat Fill depth presets - minimum samples per percussion folder
# More samples = more variety for the Beat Fill algorithm
BEATFILL_DEPTH = {
    "minimal": 5,   # Bare minimum for Beat Fill to work
    "normal": 10,   # Good variety without dominating the pack
    "deep": 15,     # Rich percussion selection
    "max": 20,      # Maximum percussion variety
}


def get_beatfill_minimum(max_samples: int, depth: str | None = None) -> int:
    """Calculate minimum samples per percussion folder.

    If depth is specified, use that preset.
    Otherwise, scale based on pack size:
    - Small packs (<=32): minimal (5)
    - Medium packs (33-64): normal (10)
    - Large packs (65-128): deep (15)
    - Very large packs (>128): max (20)

    Args:
        max_samples: Maximum samples in the pack.
        depth: Optional depth preset ("minimal", "normal", "deep", "max").

    Returns:
        Minimum samples per percussion folder.
    """
    if depth and depth in BEATFILL_DEPTH:
        return BEATFILL_DEPTH[depth]

    # Scale with pack size
    if max_samples <= 32:
        return BEATFILL_DEPTH["minimal"]
    elif max_samples <= 64:
        return BEATFILL_DEPTH["normal"]
    elif max_samples <= 128:
        return BEATFILL_DEPTH["deep"]
    else:
        return BEATFILL_DEPTH["max"]


# Tag vocabulary for prompt parsing
INSTRUMENT_TAGS = {
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
    "drums",
}

CHARACTER_TAGS = {
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
}

MOOD_TAGS = {
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
}

GENRE_TAGS = {
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
}

ALL_TAGS = INSTRUMENT_TAGS | CHARACTER_TAGS | MOOD_TAGS | GENRE_TAGS

# Tag expansion - generic terms expand to specific tags for matching
TAG_EXPANSIONS = {
    # Instrument categories
    "drums": ["kick", "snare", "hihat", "clap", "rim", "tom", "cymbal", "percussion"],
    "percussion": ["kick", "snare", "hihat", "clap", "rim", "tom", "cymbal", "shaker", "tambourine", "conga", "bongo", "cowbell"],
    "hats": ["hihat", "cymbal"],
    "synths": ["synth", "pad", "lead", "pluck", "stab", "chord", "arp", "keys"],
    "vocals": ["vocal", "vox", "voice", "chant", "speech"],
    "effects": ["fx", "sfx", "riser", "downlifter", "impact", "texture", "noise", "atmosphere"],
    # Genres implicitly include drums
    "hiphop": ["kick", "snare", "hihat", "808", "trap"],
    "trap": ["kick", "snare", "hihat", "808"],
    "techno": ["kick", "snare", "hihat", "percussion"],
    "house": ["kick", "snare", "hihat", "percussion"],
    "dnb": ["kick", "snare", "hihat", "break"],
    "dubstep": ["kick", "snare", "hihat", "bass", "sub"],
    "breaks": ["kick", "snare", "hihat", "break"],
}

# Creative/abstract term associations - maps evocative words to concrete tags
# These help interpret prompts like "floral dream pop" or "midnight jungle vibes"
CREATIVE_ASSOCIATIONS = {
    # Nature/organic imagery
    "floral": ["soft", "bright", "pad", "warm", "mellow"],
    "dreamy": ["pad", "ambient", "soft", "wet", "warm", "mellow"],
    "ethereal": ["pad", "ambient", "soft", "bright", "atmosphere"],
    "organic": ["acoustic", "warm", "soft", "percussion"],
    "earthy": ["acoustic", "warm", "dark", "percussion", "bass"],
    "airy": ["bright", "soft", "pad", "ambient"],
    "watery": ["wet", "ambient", "soft", "pad"],
    "jungle": ["percussion", "dark", "bass", "dnb", "atmosphere"],
    "forest": ["ambient", "dark", "soft", "texture", "atmosphere"],
    "ocean": ["ambient", "pad", "soft", "wet", "atmosphere"],
    "rain": ["ambient", "soft", "texture", "noise", "atmosphere"],
    "storm": ["dark", "aggressive", "impact", "noise", "atmosphere"],
    "fire": ["aggressive", "distorted", "dark", "energetic"],
    "ice": ["cold", "bright", "clean", "pad"],
    "crystal": ["bright", "clean", "pluck", "hihat"],
    "smoke": ["dark", "ambient", "pad", "soft", "texture"],
    "fog": ["ambient", "soft", "pad", "wet", "atmosphere"],
    # Time/mood imagery
    "midnight": ["dark", "ambient", "pad", "cold", "atmosphere"],
    "dawn": ["bright", "soft", "warm", "pad", "ambient"],
    "dusk": ["warm", "mellow", "dark", "pad", "ambient"],
    "sunset": ["warm", "mellow", "pad", "soft"],
    "golden": ["warm", "bright", "vintage", "analog"],
    "neon": ["bright", "modern", "futuristic", "electro", "synth"],
    "nostalgic": ["vintage", "warm", "lo-fi", "analog", "retro"],
    "melancholy": ["dark", "melancholic", "pad", "soft", "cold"],
    "euphoric": ["bright", "uplifting", "energetic", "pad"],
    "haunted": ["dark", "eerie", "haunting", "ambient", "atmosphere"],
    "dreampop": ["pad", "ambient", "soft", "bright", "wet"],
    "shoegaze": ["pad", "distorted", "wet", "noise", "texture"],
    # Texture/feeling
    "fuzzy": ["distorted", "saturated", "warm", "lo-fi"],
    "crispy": ["punchy", "tight", "bright", "hihat"],
    "crunchy": ["distorted", "saturated", "punchy"],
    "smooth": ["soft", "clean", "warm", "pad"],
    "gritty": ["distorted", "dirty", "dark", "industrial"],
    "silky": ["soft", "clean", "warm", "pad", "bright"],
    "hazy": ["lo-fi", "soft", "wet", "ambient", "warm"],
    "dusty": ["lo-fi", "vintage", "warm", "dirty"],
    "metallic": ["cold", "industrial", "hihat", "percussion"],
    "glassy": ["bright", "clean", "pluck", "cold"],
    "velvet": ["soft", "warm", "pad", "dark"],
    "rough": ["distorted", "dirty", "aggressive", "hard"],
    # Genre/style associations (not in core vocabulary)
    "pop": ["bright", "punchy", "clean", "energetic", "kick", "snare", "hihat"],
    "rock": ["punchy", "distorted", "aggressive", "kick", "snare"],
    "jazz": ["acoustic", "soft", "warm", "percussion", "mellow"],
    "classical": ["acoustic", "soft", "pad", "warm"],
    "soul": ["warm", "soft", "analog", "vintage", "bass"],
    "funk": ["punchy", "warm", "bass", "percussion", "tight"],
    "rnb": ["warm", "soft", "bass", "pad", "808"],
    "lofi": ["lo-fi", "warm", "vintage", "soft", "dusty"],
    "chillwave": ["pad", "ambient", "warm", "soft", "wet", "retro"],
    "synthwave": ["synth", "retro", "pad", "bass", "futuristic"],
    "vaporwave": ["retro", "lo-fi", "pad", "ambient", "warm"],
    "industrial": ["industrial", "dark", "aggressive", "distorted", "cold"],
    "glitch": ["digital", "experimental", "fx", "short"],
    "idm": ["experimental", "digital", "percussion", "fx"],
    # Energy/vibe
    "chill": ["mellow", "soft", "ambient", "warm", "calm"],
    "relaxed": ["mellow", "soft", "calm", "ambient", "warm"],
    "intense": ["aggressive", "energetic", "hard", "punchy"],
    "powerful": ["punchy", "hard", "aggressive", "bass", "impact"],
    "delicate": ["soft", "bright", "thin", "short"],
    "massive": ["fat", "wide", "bass", "sub", "impact"],
    "minimal": ["clean", "short", "tight", "dry"],
    "lush": ["pad", "wet", "wide", "layered", "ambient"],
    "sparse": ["dry", "short", "minimal", "clean"],
    "dense": ["layered", "fat", "wide", "wet"],
    "hypnotic": ["loop", "percussion", "ambient", "mellow"],
    "groovy": ["punchy", "tight", "bass", "percussion", "warm"],
    "bouncy": ["punchy", "tight", "energetic", "bass"],
    "heavy": ["bass", "sub", "dark", "aggressive", "fat"],
    "light": ["bright", "soft", "thin", "short", "clean"],
    "spacey": ["ambient", "wet", "pad", "wide", "atmosphere"],
    "cosmic": ["ambient", "pad", "futuristic", "wide", "atmosphere"],
    "psychedelic": ["wet", "wide", "experimental", "pad", "texture"],
    "tribal": ["percussion", "tom", "conga", "bongo", "shaker"],
    "urban": ["hiphop", "808", "bass", "trap"],
    "underground": ["dark", "experimental", "industrial", "bass"],
}


@dataclass
class ParsedPrompt:
    """Parsed prompt for pack generation."""

    tags: list[str] = field(default_factory=list)
    bpm_min: float | None = None
    bpm_max: float | None = None
    key: str | None = None
    exclude_tags: list[str] = field(default_factory=list)


def parse_prompt_simple(prompt: str, use_associations: bool = True) -> ParsedPrompt:
    """Parse a natural language prompt into structured query.

    This is a simple keyword-based parser. For more sophisticated parsing,
    use the AI-powered parser.

    Args:
        prompt: Natural language prompt like "dark 808 kicks, 120-140 BPM"
        use_associations: Whether to expand creative/abstract terms to concrete tags.

    Returns:
        ParsedPrompt with extracted criteria.
    """
    prompt_lower = prompt.lower()
    result = ParsedPrompt()

    # Extract tags by matching against vocabulary
    words = re.findall(r"\b[\w-]+\b", prompt_lower)
    for word in words:
        # Normalize hyphenated words (hip-hop -> hiphop, lo-fi -> lo-fi)
        normalized = word.replace("-", "")

        # Check exact match (original and normalized)
        if word in ALL_TAGS:
            result.tags.append(word)
        elif normalized in ALL_TAGS:
            result.tags.append(normalized)
        # Check singular form (remove trailing 's')
        elif word.endswith("s") and word[:-1] in ALL_TAGS:
            result.tags.append(word[:-1])
        elif normalized.endswith("s") and normalized[:-1] in ALL_TAGS:
            result.tags.append(normalized[:-1])
        # Check creative associations for abstract/evocative terms
        elif use_associations:
            if word in CREATIVE_ASSOCIATIONS:
                result.tags.extend(CREATIVE_ASSOCIATIONS[word])
            elif normalized in CREATIVE_ASSOCIATIONS:
                result.tags.extend(CREATIVE_ASSOCIATIONS[normalized])

    # Extract BPM range patterns
    # Pattern: "120-140 BPM" or "120 - 140 bpm"
    bpm_range_match = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*bpm", prompt_lower)
    if bpm_range_match:
        result.bpm_min = float(bpm_range_match.group(1))
        result.bpm_max = float(bpm_range_match.group(2))
    else:
        # Pattern: "at 128 BPM" or "128 bpm"
        bpm_single_match = re.search(r"(\d{2,3})\s*bpm", prompt_lower)
        if bpm_single_match:
            bpm = float(bpm_single_match.group(1))
            # Create a range around the single value (±5%)
            result.bpm_min = bpm * 0.95
            result.bpm_max = bpm * 1.05

    # Extract musical key
    key_patterns = [
        # "C minor", "F# major", "Bb minor"
        r"\b([A-Ga-g][#b]?)\s*(major|minor|maj|min)\b",
        # "Cm", "F#m", "Bbm" (shorthand)
        r"\b([A-Ga-g][#b]?)(m)\b",
    ]

    for pattern in key_patterns:
        key_match = re.search(pattern, prompt, re.IGNORECASE)
        if key_match:
            note = key_match.group(1).upper()
            quality = key_match.group(2).lower()

            # Normalize quality
            if quality in ("m", "min", "minor"):
                quality = "minor"
            elif quality in ("maj", "major"):
                quality = "major"

            result.key = f"{note} {quality}"
            break

    # Extract exclusions (words after "not", "without", "no")
    exclude_match = re.search(r"\b(?:not|without|no)\s+([\w\s,]+?)(?:\.|,|$)", prompt_lower)
    if exclude_match:
        exclude_words = re.findall(r"\b[\w-]+\b", exclude_match.group(1))
        result.exclude_tags = [w for w in exclude_words if w in ALL_TAGS]

    return result


def _expand_tags(tags: list[str]) -> set[str]:
    """Expand generic tags to include specific matches.

    For example, "drums" expands to include kick, snare, hihat, etc.

    Args:
        tags: List of tag names.

    Returns:
        Set of all tags including expansions.
    """
    expanded = set(tags)
    for tag in tags:
        if tag in TAG_EXPANSIONS:
            expanded.update(TAG_EXPANSIONS[tag])
    return expanded


def score_sample(
    sample: dict[str, Any],
    sample_tags: list[str],
    prompt: ParsedPrompt,
) -> float:
    """Score a sample's relevance to a parsed prompt.

    Args:
        sample: Sample dict from database.
        sample_tags: List of tag names for this sample.
        prompt: Parsed prompt criteria.

    Returns:
        Relevance score (higher is better, 0 means excluded).
    """
    score = 0.0
    sample_tags_lower = [t.lower() for t in sample_tags]

    # Check for excluded tags first (with expansion)
    if prompt.exclude_tags:
        expanded_excludes = _expand_tags(prompt.exclude_tags)
        for tag in expanded_excludes:
            if tag in sample_tags_lower:
                return 0.0  # Excluded

    # Score tag matches (with expansion)
    if prompt.tags:
        expanded_prompt_tags = _expand_tags(prompt.tags)
        matches = sum(1 for tag in expanded_prompt_tags if tag in sample_tags_lower)
        score += matches * 10  # 10 points per matching tag

    # Score BPM match
    sample_bpm = sample.get("bpm")
    if sample_bpm and prompt.bpm_min is not None and prompt.bpm_max is not None:
        if prompt.bpm_min <= sample_bpm <= prompt.bpm_max:
            score += 5  # Bonus for BPM in range
        else:
            # Penalty for BPM out of range (but don't exclude)
            distance = min(
                abs(sample_bpm - prompt.bpm_min), abs(sample_bpm - prompt.bpm_max)
            )
            score -= min(distance / 20, 3)  # Small penalty, max 3 points

    # Score key match
    sample_key = sample.get("detected_key")
    if sample_key and prompt.key:
        if sample_key.lower() == prompt.key.lower():
            score += 5  # Bonus for key match
        elif sample_key.split()[0] == prompt.key.split()[0]:
            # Same root note, different quality
            score += 2

    return max(0, score)


def select_samples_for_pack(
    samples: list[dict[str, Any]],
    tag_mapping: dict[int, list[str]],
    prompt: ParsedPrompt,
    max_samples: int = 64,
    max_size_bytes: int | None = None,
    beatfill_depth: str | None = None,
) -> list[dict[str, Any]]:
    """Select samples for a pack based on prompt.

    Prioritizes percussion folders (Kick, Snare, HiHat) to ensure sufficient
    samples for the Play+ Beat Fill algorithm.

    Args:
        samples: List of sample dicts from database.
        tag_mapping: Dict mapping sample ID to list of tag names.
        prompt: Parsed prompt criteria.
        max_samples: Maximum samples to select.
        max_size_bytes: Maximum total size in bytes (Play+ limit is ~32MB).
        beatfill_depth: Optional depth preset ("minimal", "normal", "deep", "max").
            If None, scales automatically with pack size.

    Returns:
        List of selected samples, sorted by relevance.
    """
    # Score and categorize all matching samples by folder
    scored_by_folder: dict[str, list[tuple[float, dict[str, Any]]]] = {}

    for sample in samples:
        sample_id = sample["id"]
        sample_tags = tag_mapping.get(sample_id, [])

        score = score_sample(sample, sample_tags, prompt)
        if score > 0:
            folder = get_folder_for_tags(sample_tags)
            if folder not in scored_by_folder:
                scored_by_folder[folder] = []
            scored_by_folder[folder].append((score, sample))

    # Sort each folder's samples by score
    for folder in scored_by_folder:
        scored_by_folder[folder].sort(key=lambda x: x[0], reverse=True)

    # Calculate dynamic Beat Fill minimum based on pack size or explicit depth
    perc_minimum = get_beatfill_minimum(max_samples, beatfill_depth)

    # Phase 1: Fill percussion folders to minimum for Beat Fill
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    total_size = 0

    def can_add_sample(sample: dict[str, Any]) -> bool:
        """Check if sample can be added within size limit."""
        if max_size_bytes is None:
            return True
        sample_size = sample.get("file_size_bytes") or 0
        return total_size + sample_size <= max_size_bytes

    def add_sample(sample: dict[str, Any]) -> bool:
        """Add sample if within limits. Returns True if added."""
        nonlocal total_size
        if len(selected) >= max_samples:
            return False
        if not can_add_sample(sample):
            return False
        if sample["id"] in used_ids:
            return False
        selected.append(sample)
        used_ids.add(sample["id"])
        total_size += sample.get("file_size_bytes") or 0
        return True

    for folder in PERCUSSION_FOLDERS:
        if folder in scored_by_folder:
            count = 0
            for score, sample in scored_by_folder[folder]:
                if count >= perc_minimum:
                    break
                if len(selected) >= max_samples:
                    break
                if add_sample(sample):
                    count += 1

    # Phase 2: Weighted allocation for remaining slots
    # Allocate proportionally based on folder weights
    if len(selected) < max_samples:
        remaining_slots = max_samples - len(selected)

        # Calculate quota per folder based on weights
        # Only include folders that have remaining samples
        available_folders = []
        for folder in INSTRUMENT_FOLDERS:
            if folder in scored_by_folder:
                remaining_count = sum(
                    1 for _, s in scored_by_folder[folder] if s["id"] not in used_ids
                )
                if remaining_count > 0:
                    available_folders.append(folder)

        if available_folders:
            # Calculate total weight of available folders
            total_weight = sum(FOLDER_WEIGHTS.get(f, 1) for f in available_folders)

            # Calculate quotas (at least 1 slot per folder if available)
            folder_quotas: dict[str, int] = {}
            allocated = 0
            for folder in available_folders:
                weight = FOLDER_WEIGHTS.get(folder, 1)
                quota = max(1, int(remaining_slots * weight / total_weight))
                folder_quotas[folder] = quota
                allocated += quota

            # Distribute any remaining slots to highest-priority folders
            while allocated < remaining_slots:
                for folder in sorted(
                    available_folders, key=lambda f: FOLDER_WEIGHTS.get(f, 1), reverse=True
                ):
                    if allocated >= remaining_slots:
                        break
                    folder_quotas[folder] += 1
                    allocated += 1

            # Fill each folder up to its quota
            for folder in sorted(
                available_folders, key=lambda f: FOLDER_WEIGHTS.get(f, 1), reverse=True
            ):
                quota = folder_quotas.get(folder, 0)
                added_count = 0

                for score, sample in scored_by_folder[folder]:
                    if added_count >= quota:
                        break
                    if len(selected) >= max_samples:
                        break
                    if max_size_bytes and total_size >= max_size_bytes:
                        break
                    if add_sample(sample):
                        added_count += 1

        # Phase 3: Fill any remaining slots with highest scoring samples overall
        # (handles case where some folders couldn't fill their quotas)
        if len(selected) < max_samples:
            remaining: list[tuple[float, dict[str, Any]]] = []
            for folder, samples_list in scored_by_folder.items():
                for score, sample in samples_list:
                    if sample["id"] not in used_ids:
                        remaining.append((score, sample))

            # Sort by score descending
            remaining.sort(key=lambda x: x[0], reverse=True)

            for score, sample in remaining:
                if len(selected) >= max_samples:
                    break
                if max_size_bytes and total_size >= max_size_bytes:
                    break
                add_sample(sample)

    return selected


def parse_prompt_with_ai(prompt: str, api_key: str | None = None) -> ParsedPrompt:
    """Parse a natural language prompt using AI for interpretation.

    Uses Claude to interpret creative/abstract prompts into concrete tags
    from the vocabulary. Falls back to simple parsing if AI is unavailable.

    Args:
        prompt: Natural language prompt like "floral dream pop" or "midnight jungle vibes"
        api_key: Anthropic API key (uses environment variable if None)

    Returns:
        ParsedPrompt with extracted criteria.
    """
    # First get simple parsing results as a baseline
    simple_result = parse_prompt_simple(prompt, use_associations=True)

    # Try AI interpretation
    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception:
        return simple_result

    # Build vocabulary string for the prompt
    all_tags_list = sorted(ALL_TAGS)
    vocab_str = ", ".join(all_tags_list)

    ai_prompt = f"""You are helping select audio samples for a music production pack.

The user wants: "{prompt}"

Interpret this creative description and select the most relevant tags from this vocabulary:
{vocab_str}

Consider:
- What instruments would fit this aesthetic?
- What sonic character/texture is implied?
- What mood does this evoke?
- What genre associations does this have?

Return ONLY a JSON object with these fields:
- "tags": list of 5-15 tags from the vocabulary above that best match the creative intent
- "bpm_min": optional minimum BPM if the description implies tempo (null if not)
- "bpm_max": optional maximum BPM (null if not)
- "key": optional musical key like "C minor" if implied (null if not)
- "reasoning": brief explanation of your interpretation

Only include tags from the vocabulary. Be generous but relevant - the user wants samples that evoke the feeling they described."""

    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": ai_prompt}],
        )

        if not response.content or len(response.content) == 0:
            return simple_result

        response_text = response.content[0].text

        # Parse JSON from response
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if not json_match:
            return simple_result

        data = json.loads(json_match.group())

        # Build ParsedPrompt from AI response
        ai_tags = data.get("tags", [])
        # Filter to only valid tags
        valid_ai_tags = [tag.lower() for tag in ai_tags if tag.lower() in ALL_TAGS]

        # Combine with simple parsing results (AI tags take priority but we keep unique simple tags)
        combined_tags = list(valid_ai_tags)
        for tag in simple_result.tags:
            if tag not in combined_tags:
                combined_tags.append(tag)

        result = ParsedPrompt(
            tags=combined_tags,
            bpm_min=data.get("bpm_min") or simple_result.bpm_min,
            bpm_max=data.get("bpm_max") or simple_result.bpm_max,
            key=data.get("key") or simple_result.key,
            exclude_tags=simple_result.exclude_tags,
        )

        return result

    except Exception:
        return simple_result


def generate_pack_name(prompt: ParsedPrompt) -> str:
    """Generate a pack name from parsed prompt.

    Args:
        prompt: Parsed prompt.

    Returns:
        Generated pack name (filesystem safe).
    """
    parts = []

    # Add up to 3 tags
    for tag in prompt.tags[:3]:
        # Sanitize tag
        clean_tag = re.sub(r"[^\w-]", "", tag)
        if clean_tag:
            parts.append(clean_tag.capitalize())

    # Add BPM if specified
    if prompt.bpm_min and prompt.bpm_max:
        if prompt.bpm_min == prompt.bpm_max:
            parts.append(f"{int(prompt.bpm_min)}BPM")
        else:
            parts.append(f"{int(prompt.bpm_min)}-{int(prompt.bpm_max)}")

    # Add key if specified
    if prompt.key:
        key_clean = prompt.key.replace(" ", "_").replace("#", "sharp")
        parts.append(key_clean)

    if not parts:
        parts = ["Pack"]

    name = "_".join(parts)

    # Ensure filesystem safe
    name = re.sub(r'[<>:"/\\|?*]', "", name)

    return name
