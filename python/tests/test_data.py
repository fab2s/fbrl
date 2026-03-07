"""Tests for fbrl/data.py — constants and font discovery."""
import string
import pytest

from fbrl.data import BIGRAMS_200, WORDS_200, discover_fonts


class TestBigrams200:
    def test_length(self):
        assert len(BIGRAMS_200) == 200

    def test_all_two_char_lowercase(self):
        for bg in BIGRAMS_200:
            assert len(bg) == 2, f"'{bg}' is not 2 chars"
            assert bg.islower(), f"'{bg}' is not lowercase"

    def test_no_duplicates(self):
        assert len(set(BIGRAMS_200)) == 200

    def test_all_letters_covered(self):
        chars = set(''.join(BIGRAMS_200))
        assert chars == set(string.ascii_lowercase)


class TestWords200:
    def test_length(self):
        assert len(WORDS_200) == 200

    def test_all_four_char_lowercase(self):
        for w in WORDS_200:
            assert len(w) == 4, f"'{w}' is not 4 chars"
            assert w.islower(), f"'{w}' is not lowercase"

    def test_no_duplicates(self):
        assert len(set(WORDS_200)) == 200

    def test_all_letters_covered(self):
        chars = set(''.join(WORDS_200))
        assert chars == set(string.ascii_lowercase)


class TestDiscoverFonts:
    def test_default(self):
        result = discover_fonts('default')
        assert result == [('default', None)]
