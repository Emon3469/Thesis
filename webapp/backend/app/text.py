"""Text handling: the exact thesis tokenizer and the Bangla / Banglish input rule.

The tokenizer is not re-implemented: it is executed from eval/w2v_kaggle_dataset/tokenize.py, the same file every
thesis evaluation used, so a sentence is split here exactly as it was split when the model was scored.
"""
import unicodedata

import config as C

_ns: dict = {}
exec((C.EVAL_DATA / "tokenize.py").read_text(encoding="utf-8"), _ns)
tokenize = _ns["tokenize"]

ZERO_WIDTH = "​‌‍﻿"
MAX_WORD_CHARS = 40
MAX_SENTENCE_CHARS = 600
MAX_SENTENCES = 30
MAX_WORDS = 12


class InputError(ValueError):
    """Raised for input that is not Bangla or Banglish; the message is shown to the user as is."""


def is_bengali_char(c: str) -> bool:
    return "ঀ" <= c <= "৿"


def is_latin_letter(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def char_ok(c: str) -> bool:
    """Bangla script, plain Latin letters (Banglish), digits, whitespace, punctuation and zero-width joiners."""
    if is_bengali_char(c) or is_latin_letter(c) or c.isspace() or c in ZERO_WIDTH or ("0" <= c <= "9"):
        return True
    return unicodedata.category(c)[0] == "P"


def check_text(text: str, max_chars: int, what: str) -> str:
    if not isinstance(text, str):
        raise InputError(f"{what} must be text.")
    t = unicodedata.normalize("NFC", text).strip()
    if not t:
        raise InputError(f"{what} is empty.")
    if len(t) > max_chars:
        raise InputError(f"{what} is longer than {max_chars} characters.")
    bad = sorted({c for c in t if not char_ok(c)})
    if bad:
        shown = " ".join(bad[:8])
        raise InputError(f"Only Bangla (বাংলা) or Banglish (Latin letters) are accepted. Not allowed in {what}: {shown}")
    if not any(is_bengali_char(c) or is_latin_letter(c) for c in t):
        raise InputError(f"{what} has no Bangla or Banglish letters.")
    return t


def token_kind(tok: str) -> str:
    """'bangla' (has Bangla letters), 'banglish' (Latin letters only), 'number' or 'other'."""
    if any(is_bengali_char(c) for c in tok):
        return "bangla"
    if any(is_latin_letter(c) for c in tok) and all(is_latin_letter(c) for c in tok):
        return "banglish"
    if all(c.isdigit() for c in tok):
        return "number"
    return "other"
