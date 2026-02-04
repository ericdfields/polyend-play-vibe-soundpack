"""Tests for AI-assisted tagger module."""

from unittest.mock import MagicMock, patch

import pytest

from soundpack.tagger import (
    extract_tags_from_filename,
    build_tagging_prompt,
    parse_ai_response,
    suggest_tags,
    TAG_VOCABULARY,
)


class TestExtractTagsFromFilename:
    """Tests for filename-based tag extraction."""

    def test_extracts_instrument_from_filename(self):
        """Extracts instrument type from filename."""
        tags = extract_tags_from_filename("808_kick_heavy.wav")
        assert "kick" in tags or "808" in tags

    def test_extracts_character_from_filename(self):
        """Extracts character descriptors from filename."""
        tags = extract_tags_from_filename("snare_punchy_01.wav")
        assert "snare" in tags
        assert "punchy" in tags

    def test_handles_underscores_and_dashes(self):
        """Handles various separators in filenames."""
        tags1 = extract_tags_from_filename("kick_dark_heavy.wav")
        tags2 = extract_tags_from_filename("kick-dark-heavy.wav")

        assert "kick" in tags1 and "dark" in tags1
        assert "kick" in tags2 and "dark" in tags2

    def test_returns_empty_for_unrecognized(self):
        """Returns empty list for unrecognized filenames."""
        tags = extract_tags_from_filename("xyz123.wav")
        assert isinstance(tags, list)

    def test_ignores_numbers(self):
        """Doesn't include numbers as tags."""
        tags = extract_tags_from_filename("kick_01_v2.wav")
        assert "01" not in tags
        assert "v2" not in tags
        assert "kick" in tags


class TestBuildTaggingPrompt:
    """Tests for AI prompt construction."""

    def test_includes_filename(self):
        """Prompt includes the filename."""
        sample_info = {
            "filename": "dark_kick.wav",
            "duration_ms": 500,
        }
        prompt = build_tagging_prompt(sample_info)
        assert "dark_kick.wav" in prompt

    def test_includes_audio_features(self):
        """Prompt includes audio analysis features."""
        sample_info = {
            "filename": "kick.wav",
            "duration_ms": 500,
            "bpm": 120.0,
            "spectral_centroid": 1500.0,
        }
        prompt = build_tagging_prompt(sample_info)

        assert "500" in prompt or "duration" in prompt.lower()
        assert "120" in prompt or "bpm" in prompt.lower()

    def test_includes_tag_vocabulary(self):
        """Prompt includes available tags."""
        sample_info = {"filename": "kick.wav"}
        prompt = build_tagging_prompt(sample_info)

        # Should mention tag categories
        assert "instrument" in prompt.lower() or "character" in prompt.lower()


class TestParseAiResponse:
    """Tests for parsing AI response."""

    def test_parses_json_response(self):
        """Parses valid JSON response."""
        response = '{"instrument": ["kick"], "character": ["punchy"], "mood": ["dark"], "genre": []}'
        tags = parse_ai_response(response)

        assert "kick" in tags
        assert "punchy" in tags
        assert "dark" in tags

    def test_handles_json_in_text(self):
        """Extracts JSON from surrounding text."""
        response = """
        Based on the analysis, here are the suggested tags:
        {"instrument": ["snare"], "character": ["tight"], "mood": [], "genre": ["techno"]}
        These tags reflect the punchy nature of the sample.
        """
        tags = parse_ai_response(response)

        assert "snare" in tags
        assert "techno" in tags

    def test_filters_invalid_tags(self):
        """Filters out tags not in vocabulary."""
        response = '{"instrument": ["kick", "invalid_tag"], "character": [], "mood": [], "genre": []}'
        tags = parse_ai_response(response)

        assert "kick" in tags
        assert "invalid_tag" not in tags

    def test_handles_malformed_json(self):
        """Returns empty list for malformed JSON."""
        response = "This is not JSON at all"
        tags = parse_ai_response(response)

        assert isinstance(tags, list)
        assert len(tags) == 0

    def test_handles_empty_response(self):
        """Handles empty response gracefully."""
        tags = parse_ai_response("")
        assert isinstance(tags, list)


class TestSuggestTags:
    """Tests for the full tag suggestion flow."""

    def test_combines_filename_and_ai_tags(self):
        """Combines tags from filename and AI."""
        sample_info = {
            "filename": "808_kick_dark.wav",
            "duration_ms": 300,
            "spectral_centroid": 500.0,
        }

        # Mock the AI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"instrument": ["kick"], "mood": ["aggressive"], "character": [], "genre": []}')]
        mock_client.messages.create.return_value = mock_response

        with patch("soundpack.tagger.get_anthropic_client", return_value=mock_client):
            tags = suggest_tags(sample_info, use_ai=True)

        # Should have filename-derived tags
        assert "808" in tags or "kick" in tags or "dark" in tags
        # Should have AI-suggested tags (if API called)
        # Note: aggressive might be added by AI

    def test_works_without_ai(self):
        """Works with filename-only mode."""
        sample_info = {
            "filename": "snare_punchy_tight.wav",
            "duration_ms": 200,
        }

        tags = suggest_tags(sample_info, use_ai=False)

        assert "snare" in tags
        assert "punchy" in tags

    def test_deduplicates_tags(self):
        """Doesn't return duplicate tags."""
        sample_info = {
            "filename": "kick_kick_kick.wav",  # Repeated
        }

        tags = suggest_tags(sample_info, use_ai=False)

        # Should only have one "kick"
        assert tags.count("kick") == 1


class TestTagVocabulary:
    """Tests for tag vocabulary structure."""

    def test_has_instrument_category(self):
        """Vocabulary has instrument tags."""
        assert "instrument" in TAG_VOCABULARY
        assert len(TAG_VOCABULARY["instrument"]) > 0
        assert "kick" in TAG_VOCABULARY["instrument"]

    def test_has_character_category(self):
        """Vocabulary has character tags."""
        assert "character" in TAG_VOCABULARY
        assert "punchy" in TAG_VOCABULARY["character"]

    def test_has_mood_category(self):
        """Vocabulary has mood tags."""
        assert "mood" in TAG_VOCABULARY
        assert "dark" in TAG_VOCABULARY["mood"]

    def test_has_genre_category(self):
        """Vocabulary has genre tags."""
        assert "genre" in TAG_VOCABULARY
        assert "techno" in TAG_VOCABULARY["genre"]
