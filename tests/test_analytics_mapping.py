"""Tests for analytics/mapping.py, and the committed mapping against the committed values."""
import csv

import pytest

from analytics import mapping

CC = "Credit card"
GOOD = [
    [CC, "Getting a credit card", "Card opened without my consent or knowledge", "yes",
     "new_account", "no", "Opened in the consumer's name."],
    [CC, "Problem with fraud alerts or security freezes", "", "no", "", "yes",
     "About fraud tools."],
    [CC, "Fees or interest", "Problem with fees", "no", "", "no", "Not fraud."],
]


def write(path, rows, cols=mapping.COLUMNS):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)
    return str(path)


def write_values(path, keys):
    rows = [list(k) + ["10", "4"] for k in keys]
    return write(path, rows, ["product", "issue", "sub_issue", "complaints", "narratives"])


def test_good_mapping_loads_and_covers(tmp_path):
    m = mapping.load(write(tmp_path / "m.csv", GOOD))
    assert m[(CC, "Problem with fraud alerts or security freezes", "")]["borderline"] == "yes"
    counts = mapping.check_coverage(m, write_values(tmp_path / "v.csv", m))
    assert sum(counts.values()) == 30


@pytest.mark.parametrize("change, message", [
    ((0, 3, "maybe"), "fraud must be yes or no"),
    ((0, 4, ""), "needs a kind"),
    ((0, 4, "scam"), "needs a kind"),
    ((2, 4, "new_account"), "kind must be empty"),
    ((1, 5, "y"), "borderline must be yes or no"),
    ((2, 6, "  "), "reason is empty"),
])
def test_bad_rows_refused(tmp_path, change, message):
    rows = [list(r) for r in GOOD]
    i, j, v = change
    rows[i][j] = v
    with pytest.raises(mapping.MappingError, match=message):
        mapping.load(write(tmp_path / "m.csv", rows))


def test_duplicate_and_columns_refused(tmp_path):
    with pytest.raises(mapping.MappingError, match="appears twice"):
        mapping.load(write(tmp_path / "m.csv", GOOD + [GOOD[0]]))
    with pytest.raises(mapping.MappingError, match="columns must be"):
        mapping.load(write(tmp_path / "m2.csv", GOOD, mapping.COLUMNS[:-1] + ["why"]))


def test_coverage_both_ways(tmp_path):
    m = mapping.load(write(tmp_path / "m.csv", GOOD))
    keys = list(m)[:2] + [(CC, "New issue", "")]
    with pytest.raises(mapping.MappingError, match="not mapped"):
        mapping.check_coverage(m, write_values(tmp_path / "v.csv", keys))


def test_committed_mapping_covers_the_committed_values():
    m = mapping.load()
    counts = mapping.check_coverage(m)
    assert len(m) == 103 and sum(counts.values()) == 125220
    fraud = {k: n for k, n in counts.items() if m[k]["fraud"] == "yes"}
    assert len(fraud) == 5 and sum(fraud.values()) == 18190


def test_upload_schema_matches_the_columns():
    import json
    with open("analytics/mapping_schema.json", encoding="utf-8") as f:
        assert [c["name"] for c in json.load(f)] == mapping.COLUMNS
