"""Narrative text for code, never for the agent's planning model.

Decision (session 5): the planner never sees consumer text. The text sits in
the table narrative_text in its own file, data/agent_text.sqlite, which the
agent's database doesn't contain and its connection can't attach (see
agent/semantic.py). The agent can still filter by words through
text_has(complaint_id, phrase), which returns only 1, 0 or NULL. Code that
must read text (the tagger, the report's quotes) uses semantic.connect_text().

readable() undoes one quirk of CFPB's CSV at read time: some narratives are
stored as a Python bytes literal, b'...' with escapes such as \\n written out.
The stored text is never changed. Inside these, redactions often appear as
\\XXXX, which Python warns is an invalid escape; the warning is silenced and
the backslash kept (216 such narratives on the real data, all decoded, S5).
"""
import ast
import re
import sqlite3
import warnings
from collections import OrderedDict

TEXT_TABLE = "narrative_text"
MAX_PHRASE = 60
PHRASE_CACHE = 32          # phrases whose matching IDs are kept in memory

_BYTES_LITERAL = re.compile(r"^b(['\"]).*\1$", re.S)
_PHRASE_OK = re.compile(r"^[a-z0-9' *-]+$")
_ESCAPES = {"\\n": "\n", "\\r": "\n", "\\t": " ", "\\'": "'", '\\"': '"', "\\\\": "\\"}


class TextError(Exception):
    pass


def is_bytes_literal(text):
    return text is not None and bool(_BYTES_LITERAL.match(text))


def decode_bytes_literal(text):
    """The bytes a b'...' narrative stands for, or None if it isn't a valid
    literal. Python's warning about invalid escapes such as \\X is silenced."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, bytes) else None


def readable(text):
    """The narrative as a person would read it. A b'...' wrapper is decoded as
    UTF-8 bytes; if it isn't a valid literal (cut short, say), the common
    escapes are replaced instead. Other text is returned unchanged."""
    if not is_bytes_literal(text):
        return text
    value = decode_bytes_literal(text)
    if value is not None:
        return value.decode("utf-8", errors="replace")
    body = text[2:-1]
    return re.sub(r"\\[nrt'\"\\]", lambda m: _ESCAPES[m.group(0)], body)


def phrase_pattern(phrase):
    """Compiled pattern for a text_has phrase, or TextError saying what to fix.
    It is matched against lower-cased text, so matching is case-insensitive.

    Words match whole, with any spacing or line break between them. A word
    ending in * matches any ending (refund* finds refund, refunds, refunded).
    An apostrophe matches a straight or curly one."""
    if not isinstance(phrase, str):
        raise TextError("text_has: the phrase must be text in quotes, such as 'late fee'")
    p = " ".join(phrase.lower().split())
    if not p:
        raise TextError("text_has: the phrase is empty")
    if len(p) > MAX_PHRASE:
        raise TextError(f"text_has: the phrase is over {MAX_PHRASE} characters; use a shorter one")
    if not _PHRASE_OK.match(p):
        raise TextError("text_has: use letters, digits, spaces, apostrophes, hyphens and a "
                        "trailing * only (no other punctuation, no regular expressions)")
    words = p.split(" ")
    parts = []
    for w in words:
        star = w.endswith("*")
        core = w.rstrip("*")
        if "*" in core or not re.search(r"[a-z0-9]", core):
            raise TextError(f"text_has: {w!r} is not a word; * may only end a word")
        part = re.escape(core).replace("'", "['\u2019]")
        parts.append(part + ("[a-z0-9]*" if star else ""))
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])")


def must_contain(phrase):
    """The longest piece of the phrase that every match contains literally:
    a quick substring check before the pattern runs."""
    pieces = [piece for w in " ".join(phrase.lower().split()).split(" ")
              for piece in w.rstrip("*").split("'")]
    return max(pieces, key=len)


def open_text(text_uri):
    """Read-only connection to the text file. Never give it to the agent."""
    conn = sqlite3.connect(text_uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


class TextIndex:
    """Backs text_has() on the agent's connection.

    The first call loads every narrative once, decoded and lower-cased, from
    the text file on its own connection, and keeps it in memory (about the size
    of the text, 60 to 80 MB on the real data). The first call with a phrase
    scans that list once (a substring check, then the pattern) and keeps the
    set of matching IDs; later rows are a set lookup. Errors go to on_error
    (the guard's last_denial) as well as being raised, because SQLite replaces
    a function's exception with a generic message."""

    def __init__(self, text_uri, on_error):
        self.uri = text_uri
        self.on_error = on_error
        self.ids = None
        self.texts = None
        self.cache = OrderedDict()

    def _load(self):
        conn = open_text(self.uri)
        try:
            self.texts = [(cid, readable(text).lower()) for cid, text in conn.execute(
                f"SELECT complaint_id, narrative FROM {TEXT_TABLE}")]
        finally:
            conn.close()
        self.ids = {cid for cid, _ in self.texts}

    @staticmethod
    def phrase_pattern_key(phrase):
        return phrase_pattern(phrase).pattern

    def matches(self, phrase):
        pattern = phrase_pattern(phrase)
        key = pattern.pattern
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.texts is None:
            self._load()
        piece = must_contain(phrase)
        found = frozenset(cid for cid, text in self.texts
                          if piece in text and pattern.search(text))
        self.cache[key] = found
        if len(self.cache) > PHRASE_CACHE:
            self.cache.popitem(last=False)
        return found

    def __call__(self, complaint_id, phrase):
        try:
            found = self.matches(phrase)
        except TextError as e:
            self.on_error(str(e))
            raise
        if complaint_id not in self.ids:
            return None
        return 1 if complaint_id in found else 0
