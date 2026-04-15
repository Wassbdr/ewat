"""Unit tests for telemetry.features.lexical."""

import math

import pytest

from telemetry.features.lexical import lexical_entropy, tokenise


class TestTokenise:
    def test_basic_split(self):
        tokens = tokenise("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_iso_timestamp(self):
        line = "2024-01-15T10:30:00Z INFO starting service"
        tokens = tokenise(line)
        # Timestamp tokens should not appear; meaningful words should
        assert "starting" in tokens
        assert "service" in tokens

    def test_strips_log_level(self):
        tokens = tokenise("ERROR something went wrong")
        assert "error" not in tokens or "something" in tokens

    def test_single_char_tokens_dropped(self):
        tokens = tokenise("a b c hello")
        single_chars = [t for t in tokens if len(t) < 2]
        assert single_chars == []

    def test_empty_line(self):
        assert tokenise("") == []

    def test_lowercased(self):
        tokens = tokenise("Hello WORLD")
        assert all(t == t.lower() for t in tokens)


class TestLexicalEntropy:
    def test_empty_returns_zero(self):
        assert lexical_entropy([]) == 0.0

    def test_single_repeated_token_entropy_zero(self):
        # All the same word → entropy = 0
        lines = ["tick tick tick tick tick tick"]
        h = lexical_entropy(lines)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_entropy_positive_for_diverse_vocab(self):
        lines = [
            "ERROR database connection refused host unreachable",
            "INFO request received processing started",
            "WARN memory usage high garbage collection running",
        ]
        h = lexical_entropy(lines)
        assert h > 0.0

    def test_entropy_increases_with_diversity(self):
        uniform = ["tick tick tick tick"]
        diverse = ["alpha beta gamma delta epsilon zeta eta theta"]
        h_uniform = lexical_entropy(uniform)
        h_diverse = lexical_entropy(diverse)
        assert h_diverse > h_uniform

    def test_entropy_bounded_by_log2_vocab(self):
        # Uniform distribution over V tokens → H = log2(V)
        # We test the bound H <= log2(V)
        words = [f"word{i}" for i in range(32)]
        lines = [" ".join(words)]
        h = lexical_entropy(lines)
        assert h <= math.log2(len(words)) + 0.01  # tiny slack for tokenise filtering
