"""Text helpers used when reading narratives. The stored text is never changed.

fingerprint() gives the same value to narratives that are the same apart from
redactions, amounts, numbers, case and punctuation. Template complaints that
are copied almost word for word share a fingerprint. Narratives that differ
by an added or removed sentence do not, so counts based on it are a floor.
"""
import hashlib
import re

_AMOUNT = re.compile(r"\{\$[^}]*\}")                           # {$35.00}
_REDACTION = re.compile(r"\bx{2,}(?:/(?:x{2,}|year>))*", re.I)  # XXXX, XX/XX/XXXX, XX/XX/year>
_DIGITS = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^a-z<>]+")


def normalize(text):
    t = _AMOUNT.sub(" <amt> ", text.lower())
    t = _REDACTION.sub(" <x> ", t)
    t = _DIGITS.sub(" <n> ", t)
    return " ".join(_NON_WORD.sub(" ", t).split())


def fingerprint(text):
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]
