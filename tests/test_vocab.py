"""Tamil vocabulary and Tamil/Indic syllable validation."""

import pytest

from indra_ocr.vocab import Vocab, build_tamil_tokens, indic_valid


def test_tamil_vocab_covers_the_block():
    v = Vocab.tamil()
    for c in "க ழ ஷ ஹ ஃ ஂ ா ெ ொ ் ௨ ௱ ஔ ஓம்".replace(" ", ""):
        assert c in v.index, f"missing {c!r} ({hex(ord(c))})"
    # ASCII digits and Latin punctuation ride along from the latin base.
    for c in "0123456789.,;:!?-()%":
        assert c in v.index
    # No LaTeX command tokens: a Tamil recognizer stays character-level.
    assert not v.has_commands


def test_tamil_vocab_round_trip():
    v = Vocab.tamil()
    for line in (
        "யாதும் ஊரே யாவரும் கேளிர்",
        "செம்மொழித் தமிழ்: 18,000 கோடி (2025-26)",
        "ஸ்ரீ க்ஷேத்திரம்",
    ):
        assert v.decode(v.encode(line)) == line


def test_tamil_vocab_rejects_unknown_scripts():
    v = Vocab.tamil()
    assert v.encode("தமிழ்") is not None
    assert v.encode("ગુજરાતી") is None  # Gujarati is outside the charset
    assert "中" in v.oov("தமிழ் 中 மொழி")


def test_named_vocab():
    assert Vocab.named("tamil").chars == build_tamil_tokens()
    with pytest.raises(ValueError):
        Vocab.named("klingon")


def test_tamil_valid_lines_pass():
    for line in (
        "தமிழ்",
        "க்ஷேத்திரம்",  # conjunct: virama between two consonants
        "ஸ்ரீ",
        "கஃத",  # visarga is a letter (Lo)
        "யாதும் ஊரே",
        "கஂ",  # anusvara after a consonant
    ):
        assert indic_valid(line), line


def test_broken_tamil_lines_fail():
    for line in (
        "க்ா",  # vowel sign after the virama
        "்க",  # virama with no base
        "காி",  # two vowel signs in a row
        "ஂக",  # anusvara with no base
    ):
        assert not indic_valid(line), line


def test_latin_and_devanagari_still_validated():
    assert indic_valid("किरण") and indic_valid("হ্যালো")
    assert not indic_valid("िामथ") and not indic_valid("का ा")
    # Latin-only text never breaks the rule.
    assert indic_valid("UPI transactions: 18,000 crore")
