"""Tests for the LLM Verifier response parsers (no model calls)."""
import pytest
from src.llm_verifier import LLMVerifier


@pytest.fixture
def verifier():
    # The constructor does not contact Ollama; only verify*() calls do.
    return LLMVerifier(provider="ollama", model="qwen2.5-coder:7b")


class TestParseBinary:
    """Legacy binary YES/NO parser."""

    def test_valid_json_parsed(self, verifier):
        parsed = verifier._parse_binary('{"attempts_task": "yes", "confidence": 0.95}')
        assert parsed["attempts_task"] == "YES"
        assert parsed["confidence"] == 0.95
        assert parsed["_malformed"] is False

    def test_json_in_markdown_block(self, verifier):
        response = "```json\n{\"attempts_task\": \"NO\", \"confidence\": 0.8}\n```"
        parsed = verifier._parse_binary(response)
        assert parsed["attempts_task"] == "NO"
        assert parsed["confidence"] == 0.8

    def test_non_json_response_falls_back(self, verifier):
        parsed = verifier._parse_binary("The code attempts the task.")
        assert parsed["attempts_task"] == "YES"
        assert parsed["confidence"] == 0.0
        assert parsed["_malformed"] is True


class TestParseCov:
    """Chain-of-Verification 3-way parser."""

    def test_valid_json_parsed(self, verifier):
        response = ('{"verdict": "hallucination", "confidence": 0.9, '
                    '"fabricated": true, "justification": "fake module"}')
        parsed = verifier._parse_cov(response)
        assert parsed["verdict"] == "HALLUCINATION"
        assert parsed["confidence"] == 0.9
        assert parsed["fabricated"] is True
        assert parsed["justification"] == "fake module"
        assert parsed["_malformed"] is False

    def test_json_in_markdown_block(self, verifier):
        response = "```json\n{\"verdict\": \"VALID\", \"confidence\": 0.85}\n```"
        parsed = verifier._parse_cov(response)
        assert parsed["verdict"] == "VALID"
        assert parsed["confidence"] == 0.85
        assert parsed["fabricated"] is False

    def test_unknown_verdict_maps_to_grounded_error(self, verifier):
        parsed = verifier._parse_cov('{"verdict": "MAYBE", "confidence": 0.5}')
        assert parsed["verdict"] == "GROUNDED_ERROR"
        assert parsed["_malformed"] is False

    def test_malformed_json_falls_back(self, verifier):
        # A non-numeric confidence raises ValueError inside the parser.
        parsed = verifier._parse_cov('{"verdict": "VALID", "confidence": "high"}')
        assert parsed["verdict"] == "GROUNDED_ERROR"
        assert parsed["confidence"] == 0.0
        assert parsed["_malformed"] is True
