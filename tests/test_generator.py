"""Tests for pack generator module."""

from unittest.mock import MagicMock, patch

import pytest

from soundpack.generator import (
    ParsedPrompt,
    parse_prompt_simple,
    score_sample,
    select_samples_for_pack,
    generate_pack_name,
    _expand_tags,
    TAG_EXPANSIONS,
    CREATIVE_ASSOCIATIONS,
)


class TestParsePromptSimple:
    """Tests for simple prompt parsing (without AI)."""

    def test_extracts_instrument_tags(self):
        """Extracts instrument keywords from prompt."""
        prompt = "808 kicks and snares"
        result = parse_prompt_simple(prompt)

        assert "kick" in result.tags or "808" in result.tags
        assert "snare" in result.tags

    def test_extracts_mood_tags(self):
        """Extracts mood keywords from prompt."""
        prompt = "dark aggressive drums"
        result = parse_prompt_simple(prompt)

        assert "dark" in result.tags or "aggressive" in result.tags

    def test_extracts_character_tags(self):
        """Extracts character keywords from prompt."""
        prompt = "punchy analog bass"
        result = parse_prompt_simple(prompt)

        assert "punchy" in result.tags or "analog" in result.tags

    def test_extracts_bpm_range(self):
        """Extracts BPM range from prompt."""
        prompt = "techno drums 120-140 BPM"
        result = parse_prompt_simple(prompt)

        assert result.bpm_min == 120
        assert result.bpm_max == 140

    def test_extracts_single_bpm(self):
        """Extracts single BPM value as range."""
        prompt = "house loops at 128 BPM"
        result = parse_prompt_simple(prompt)

        # Single BPM should create a small range around it
        assert result.bpm_min is not None
        assert result.bpm_max is not None
        assert result.bpm_min <= 128 <= result.bpm_max

    def test_extracts_key(self):
        """Extracts musical key from prompt."""
        prompt = "pads in C minor"
        result = parse_prompt_simple(prompt)

        assert result.key == "C minor"

    def test_handles_various_key_formats(self):
        """Handles different key format variations."""
        test_cases = [
            ("F# major", "F# major"),
            ("A minor", "A minor"),
            ("Dm", "D minor"),
            ("G", None),  # Just a letter isn't specific enough
        ]

        for prompt_key, expected in test_cases:
            result = parse_prompt_simple(f"samples in {prompt_key}")
            if expected:
                assert result.key is not None

    def test_returns_parsed_prompt_object(self):
        """Returns ParsedPrompt with all fields."""
        result = parse_prompt_simple("dark kicks")

        assert isinstance(result, ParsedPrompt)
        assert hasattr(result, "tags")
        assert hasattr(result, "bpm_min")
        assert hasattr(result, "bpm_max")
        assert hasattr(result, "key")
        assert hasattr(result, "exclude_tags")

    def test_parses_hyphenated_genres(self):
        """Parses hip-hop as hiphop, lo-fi as lo-fi."""
        result = parse_prompt_simple("hip-hop beats")
        assert "hiphop" in result.tags

    def test_parses_lofi_with_hyphen(self):
        """Parses lo-fi correctly."""
        result = parse_prompt_simple("lo-fi drums")
        assert "lo-fi" in result.tags

    def test_creative_associations_floral(self):
        """Creative terms like 'floral' expand to associated tags."""
        result = parse_prompt_simple("floral dream pop")
        # floral should map to soft, bright, pad, warm, mellow
        assert "soft" in result.tags or "bright" in result.tags or "pad" in result.tags

    def test_creative_associations_dreamy(self):
        """Creative term 'dreamy' expands to ambient/soft tags."""
        result = parse_prompt_simple("dreamy textures")
        # dreamy should map to pad, ambient, soft, wet, warm, mellow
        assert "pad" in result.tags or "ambient" in result.tags or "soft" in result.tags

    def test_creative_associations_midnight(self):
        """Creative term 'midnight' expands to dark/ambient tags."""
        result = parse_prompt_simple("midnight jungle vibes")
        # midnight should map to dark, ambient, pad, cold, atmosphere
        # jungle should map to percussion, dark, bass, dnb, atmosphere
        assert "dark" in result.tags

    def test_creative_associations_disabled(self):
        """Can disable creative associations."""
        result = parse_prompt_simple("floral dream pop", use_associations=False)
        # Without associations, floral/dream/pop won't match anything
        # Only explicit tags would match
        assert "soft" not in result.tags
        assert "pad" not in result.tags

    def test_creative_associations_combined_with_explicit_tags(self):
        """Creative associations work alongside explicit tags."""
        result = parse_prompt_simple("dreamy kick snare")
        # Should have explicit tags
        assert "kick" in result.tags
        assert "snare" in result.tags
        # And creative associations
        assert "pad" in result.tags or "ambient" in result.tags or "soft" in result.tags

    def test_genre_associations_pop(self):
        """Genre terms like 'pop' expand to associated tags."""
        result = parse_prompt_simple("dream pop sounds")
        # pop should map to bright, punchy, clean, energetic, kick, snare, hihat
        # dream should map to pad, ambient, soft, wet, warm, mellow
        assert "kick" in result.tags or "snare" in result.tags or "bright" in result.tags

    def test_texture_associations_hazy(self):
        """Texture terms like 'hazy' expand to lo-fi/ambient tags."""
        result = parse_prompt_simple("hazy nostalgic sounds")
        # hazy should map to lo-fi, soft, wet, ambient, warm
        # nostalgic should map to vintage, warm, lo-fi, analog, retro
        assert "lo-fi" in result.tags or "warm" in result.tags or "vintage" in result.tags


class TestScoreSample:
    """Tests for sample scoring/relevance."""

    def test_higher_score_for_more_tag_matches(self):
        """Samples with more matching tags score higher."""
        sample = {
            "id": 1,
            "filename": "kick.wav",
            "bpm": 120,
            "detected_key": "C minor",
        }
        sample_tags = ["kick", "808", "punchy"]

        prompt1 = ParsedPrompt(tags=["kick"])
        prompt2 = ParsedPrompt(tags=["kick", "808", "punchy"])

        score1 = score_sample(sample, sample_tags, prompt1)
        score2 = score_sample(sample, sample_tags, prompt2)

        assert score2 > score1

    def test_bpm_in_range_scores_higher(self):
        """Samples with BPM in requested range score higher."""
        sample_in_range = {"id": 1, "filename": "a.wav", "bpm": 120}
        sample_out_range = {"id": 2, "filename": "b.wav", "bpm": 80}

        prompt = ParsedPrompt(tags=["kick"], bpm_min=110, bpm_max=130)

        score_in = score_sample(sample_in_range, ["kick"], prompt)
        score_out = score_sample(sample_out_range, ["kick"], prompt)

        assert score_in > score_out

    def test_key_match_scores_higher(self):
        """Samples with matching key score higher."""
        sample_match = {"id": 1, "filename": "a.wav", "detected_key": "C minor"}
        sample_nomatch = {"id": 2, "filename": "b.wav", "detected_key": "F major"}

        prompt = ParsedPrompt(tags=["pad"], key="C minor")

        score_match = score_sample(sample_match, ["pad"], prompt)
        score_nomatch = score_sample(sample_nomatch, ["pad"], prompt)

        assert score_match > score_nomatch

    def test_zero_score_for_excluded_tags(self):
        """Samples with excluded tags get zero score."""
        sample = {"id": 1, "filename": "kick.wav"}
        sample_tags = ["kick", "distorted"]

        prompt = ParsedPrompt(tags=["kick"], exclude_tags=["distorted"])

        score = score_sample(sample, sample_tags, prompt)
        assert score == 0


class TestSelectSamplesForPack:
    """Tests for pack selection algorithm."""

    @pytest.fixture
    def sample_library(self):
        """Create a mock sample library."""
        return [
            {"id": 1, "filename": "kick1.wav", "bpm": 120, "detected_key": "C minor"},
            {"id": 2, "filename": "kick2.wav", "bpm": 120, "detected_key": "C minor"},
            {"id": 3, "filename": "snare1.wav", "bpm": 120, "detected_key": None},
            {"id": 4, "filename": "hihat1.wav", "bpm": 120, "detected_key": None},
            {"id": 5, "filename": "bass1.wav", "bpm": 90, "detected_key": "F major"},
        ]

    @pytest.fixture
    def tag_mapping(self):
        """Create tag mapping for samples."""
        return {
            1: ["kick", "808", "dark"],
            2: ["kick", "acoustic"],
            3: ["snare", "punchy"],
            4: ["hihat", "closed"],
            5: ["bass", "sub"],
        }

    def test_selects_matching_samples(self, sample_library, tag_mapping):
        """Selects samples that match prompt."""
        prompt = ParsedPrompt(tags=["kick"])

        selected = select_samples_for_pack(
            sample_library, tag_mapping, prompt, max_samples=10
        )

        # Should select kick samples
        selected_ids = {s["id"] for s in selected}
        assert 1 in selected_ids or 2 in selected_ids

    def test_respects_max_samples(self, sample_library, tag_mapping):
        """Respects maximum samples limit."""
        prompt = ParsedPrompt(tags=["kick", "snare", "hihat", "bass"])

        selected = select_samples_for_pack(
            sample_library, tag_mapping, prompt, max_samples=2
        )

        assert len(selected) <= 2

    def test_sorts_by_relevance(self, sample_library, tag_mapping):
        """Returns samples sorted by relevance score."""
        prompt = ParsedPrompt(tags=["kick", "808", "dark"])

        selected = select_samples_for_pack(
            sample_library, tag_mapping, prompt, max_samples=5
        )

        # First sample should be the best match (kick1 with 808 and dark)
        if len(selected) > 0:
            assert selected[0]["id"] == 1


class TestTagExpansion:
    """Tests for generic tag expansion (e.g., 'drums' -> kick, snare, hihat)."""

    def test_drums_expands_to_percussion_instruments(self):
        """'drums' tag expands to include kick, snare, hihat, etc."""
        expanded = _expand_tags(["drums"])
        assert "kick" in expanded
        assert "snare" in expanded
        assert "hihat" in expanded
        assert "drums" in expanded  # Original tag kept

    def test_non_expandable_tag_unchanged(self):
        """Tags without expansions are returned unchanged."""
        expanded = _expand_tags(["kick", "dark"])
        assert expanded == {"kick", "dark"}

    def test_drums_prompt_matches_hihat_sample(self):
        """Searching for 'drums' matches samples tagged with 'hihat'."""
        sample = {"id": 1, "filename": "hihat.wav", "bpm": 120}
        sample_tags = ["hihat", "tight"]
        prompt = ParsedPrompt(tags=["drums"])

        score = score_sample(sample, sample_tags, prompt)
        assert score > 0  # Should match via expansion

    def test_drums_prompt_matches_all_percussion(self):
        """'drums' matches kick, snare, and hihat samples."""
        samples = [
            ({"id": 1, "filename": "kick.wav"}, ["kick"]),
            ({"id": 2, "filename": "snare.wav"}, ["snare"]),
            ({"id": 3, "filename": "hihat.wav"}, ["hihat"]),
            ({"id": 4, "filename": "bass.wav"}, ["bass"]),
        ]
        prompt = ParsedPrompt(tags=["drums"])

        scores = [(s[0]["filename"], score_sample(s[0], s[1], prompt)) for s in samples]

        # Percussion should match
        assert scores[0][1] > 0  # kick
        assert scores[1][1] > 0  # snare
        assert scores[2][1] > 0  # hihat
        # Bass should not match
        assert scores[3][1] == 0

    def test_hiphop_expands_to_drums(self):
        """'hiphop' genre expands to include percussion instruments."""
        expanded = _expand_tags(["hiphop"])
        assert "kick" in expanded
        assert "snare" in expanded
        assert "hihat" in expanded
        assert "808" in expanded

    def test_genre_prompt_matches_percussion(self):
        """Genre prompts like 'hiphop' match percussion samples."""
        samples = [
            ({"id": 1, "filename": "kick.wav"}, ["kick"]),
            ({"id": 2, "filename": "snare.wav"}, ["snare"]),
            ({"id": 3, "filename": "hihat.wav"}, ["hihat"]),
        ]
        prompt = ParsedPrompt(tags=["hiphop"])

        # All percussion should match via hiphop expansion
        for sample, tags in samples:
            score = score_sample(sample, tags, prompt)
            assert score > 0, f"{sample['filename']} should match hiphop"


class TestPercussionPrioritization:
    """Tests for Beat Fill percussion folder prioritization."""

    @pytest.fixture
    def mixed_library(self):
        """Create library with mixed instrument types."""
        samples = []
        # 10 kicks
        for i in range(10):
            samples.append({"id": i, "filename": f"kick_{i}.wav", "bpm": 120})
        # 10 snares
        for i in range(10, 20):
            samples.append({"id": i, "filename": f"snare_{i}.wav", "bpm": 120})
        # 10 hihats
        for i in range(20, 30):
            samples.append({"id": i, "filename": f"hihat_{i}.wav", "bpm": 120})
        # 20 FX samples
        for i in range(30, 50):
            samples.append({"id": i, "filename": f"fx_{i}.wav", "bpm": 120})
        return samples

    @pytest.fixture
    def mixed_tag_mapping(self):
        """Create tag mapping for mixed library."""
        mapping = {}
        for i in range(10):
            mapping[i] = ["kick", "punchy"]
        for i in range(10, 20):
            mapping[i] = ["snare", "crispy"]
        for i in range(20, 30):
            mapping[i] = ["hihat", "closed"]
        for i in range(30, 50):
            mapping[i] = ["fx", "riser"]
        return mapping

    def test_prioritizes_percussion_folders(self, mixed_library, mixed_tag_mapping):
        """Ensures at least 5 samples per percussion folder before FX."""
        from soundpack.exporter import get_folder_for_tags, MIN_SAMPLES_PER_PERC_FOLDER

        # Request all types but with limited max
        prompt = ParsedPrompt(tags=["kick", "snare", "hihat", "fx"])

        # Only allow 20 samples - forces prioritization
        selected = select_samples_for_pack(
            mixed_library, mixed_tag_mapping, prompt, max_samples=20
        )

        # Count samples per folder
        folder_counts = {"Kick": 0, "Snare": 0, "HiHat": 0, "FX": 0}
        for sample in selected:
            tags = mixed_tag_mapping.get(sample["id"], [])
            folder = get_folder_for_tags(tags)
            if folder in folder_counts:
                folder_counts[folder] += 1

        # Each percussion folder should have at least 5 samples
        assert folder_counts["Kick"] >= MIN_SAMPLES_PER_PERC_FOLDER
        assert folder_counts["Snare"] >= MIN_SAMPLES_PER_PERC_FOLDER
        assert folder_counts["HiHat"] >= MIN_SAMPLES_PER_PERC_FOLDER

    def test_fills_percussion_before_fx_when_limited(self, mixed_library, mixed_tag_mapping):
        """When sample count is limited, percussion takes priority over FX."""
        from soundpack.exporter import get_folder_for_tags

        prompt = ParsedPrompt(tags=["kick", "snare", "hihat", "fx"])

        # Only 15 samples - must prioritize percussion
        selected = select_samples_for_pack(
            mixed_library, mixed_tag_mapping, prompt, max_samples=15
        )

        folder_counts = {"Kick": 0, "Snare": 0, "HiHat": 0, "FX": 0}
        for sample in selected:
            tags = mixed_tag_mapping.get(sample["id"], [])
            folder = get_folder_for_tags(tags)
            if folder in folder_counts:
                folder_counts[folder] += 1

        # Each percussion folder should have 5, leaving 0 for FX
        assert folder_counts["Kick"] == 5
        assert folder_counts["Snare"] == 5
        assert folder_counts["HiHat"] == 5
        assert folder_counts["FX"] == 0

    def test_includes_fx_when_only_fx_requested(self, mixed_library, mixed_tag_mapping):
        """FX samples are included when only FX is requested."""
        from soundpack.exporter import get_folder_for_tags

        # Only request FX - no percussion requested
        prompt = ParsedPrompt(tags=["fx", "riser"])

        selected = select_samples_for_pack(
            mixed_library, mixed_tag_mapping, prompt, max_samples=20
        )

        folder_counts = {"Kick": 0, "Snare": 0, "HiHat": 0, "FX": 0}
        for sample in selected:
            tags = mixed_tag_mapping.get(sample["id"], [])
            folder = get_folder_for_tags(tags)
            if folder in folder_counts:
                folder_counts[folder] += 1

        # Only FX samples should be selected (no percussion in prompt)
        assert folder_counts["FX"] == 20
        assert folder_counts["Kick"] == 0
        assert folder_counts["Snare"] == 0
        assert folder_counts["HiHat"] == 0


class TestGeneratePackName:
    """Tests for automatic pack name generation."""

    def test_generates_name_from_tags(self):
        """Generates name from prompt tags."""
        prompt = ParsedPrompt(tags=["808", "dark", "kick"])
        name = generate_pack_name(prompt)

        assert name  # Not empty
        assert isinstance(name, str)
        # Should include some of the tags
        name_lower = name.lower()
        assert any(tag in name_lower for tag in ["808", "dark", "kick"])

    def test_includes_bpm_in_name(self):
        """Includes BPM range in name if specified."""
        prompt = ParsedPrompt(tags=["techno"], bpm_min=130, bpm_max=140)
        name = generate_pack_name(prompt)

        # Should include BPM indicator
        assert "130" in name or "140" in name or "bpm" in name.lower()

    def test_generates_valid_filename(self):
        """Generated name is valid for filesystem."""
        prompt = ParsedPrompt(tags=["weird/chars", "special:stuff"])
        name = generate_pack_name(prompt)

        # Should not contain invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            assert char not in name
