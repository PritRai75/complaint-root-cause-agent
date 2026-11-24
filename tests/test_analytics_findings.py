"""Every number in analytics/findings.md must come from a result file (decision 57),
and analytics/derive.py must reproduce the committed derived.csv."""
import csv
import glob
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from analytics import derive

FINDINGS = "analytics/findings.md"
RESULTS = "analytics/results"
# Rule constants written in the text (thresholds, interval level, year); small
# whole numbers up to 12 count as words, as in the report's number check.
ALLOWED = {Decimal(v) for v in ("20", "25", "95", "200", "2026")}
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    """Numbers in prose: code spans and code blocks are skipped (file names, commands)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    out = []
    for tok in NUMBER.findall(text):
        tok = tok.replace(",", "")
        if tok.endswith("."):
            tok = tok[:-1]
        out.append(tok)
    return out


def candidates(results_dir=RESULTS):
    vals = set()
    for path in glob.glob(os.path.join(results_dir, "*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                for cell in row:
                    try:
                        vals.add(Decimal(cell))
                    except ArithmeticError:
                        pass
    return vals


def found(tok, vals):
    t = Decimal(tok)
    if t in ALLOWED or (t == t.to_integral_value() and 0 <= t <= 12):
        return True
    q = Decimal(1).scaleb(t.as_tuple().exponent) if "." in tok else Decimal(1)
    for v in vals:
        for x in (v, v * 100):
            if x.quantize(q, rounding=ROUND_HALF_UP) == t:
                return True
    return False


def unmatched(text, vals):
    return [tok for tok in numbers_in(text) if not found(tok, vals)]


def test_numbers_in_prose_skip_code_and_strip_commas():
    text = "A 1,689 b `q5_tally.csv` c 14.5% d\n```\n99 999\n```\n(48.9 to 54.3)."
    assert numbers_in(text) == ["1689", "14.5", "48.9", "54.3"]


def test_matching_rounds_and_reads_percentages():
    vals = {Decimal("0.145264"), Decimal("1649.2"), Decimal("187")}
    assert found("14.5", vals) and found("1649.2", vals) and found("187", vals)
    assert found("7", vals) and found("200", vals)
    assert not found("14.6", vals) and not found("188", vals) and not found("13", vals)


def test_every_number_in_findings_comes_from_a_result_file():
    with open(FINDINGS, encoding="utf-8") as f:
        text = f.read()
    missing = unmatched(text, candidates())
    assert missing == [], f"numbers not found in analytics/results/: {missing}"


def test_derive_reproduces_the_committed_derived_csv(tmp_path):
    rows = derive.derive()
    path = derive.write(rows, str(tmp_path))
    with open(path, encoding="utf-8") as a, open(os.path.join(RESULTS, derive.OUT_NAME),
                                                   encoding="utf-8") as b:
        assert a.read() == b.read()


def test_spike_rule_on_the_combined_series_matches_04_spikes():
    # Capital One's own series, where 04_spikes.sql flagged May (threshold 181.436).
    judged = dict((m, (th, sp)) for m, _, _, _, th, sp in
                  derive.spike_rule([178, 145, 151, 170, 187, 174, 185, 178]))
    th, sp = judged["2026-05-01"]
    assert th == pytest.approx(181.436, abs=5e-4) and sp
    assert sum(s for _, s in judged.values()) == 1
