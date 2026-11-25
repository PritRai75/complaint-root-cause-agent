"""Tests for analytics/export.py on a small fake database (no real data, so CI runs them)."""
import csv
import json
import sqlite3

import pytest

from analytics import export
from ingest import build_db

CC, CS = "Credit card", "Checking or savings account"


def make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(build_db.SCHEMA)
    conn.execute("INSERT INTO build_info VALUES ('built_at', '2026-11-02T20:50:00Z')")
    for i, (product, issue, sub, narr) in enumerate(rows, 1):
        conn.execute(f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) VALUES "
                     f"({', '.join('?' * len(build_db.DB_COLUMNS))})",
                     (i, "2026-0%d-15" % (1 + i % 8), product, None, issue, sub,
                      "BANK A, N.A.", None, None, "Web", "2026-01-16",
                      "Closed with explanation", "Line one\nline \"two\"", "Yes", narr,
                      "CCDB_Export_15.zip"))
    conn.commit()
    conn.close()


ROWS = [
    (CC, "Problem with a purchase shown on your statement", "Card was charged for something you did not purchase with the card", 1),
    (CC, "Problem with a purchase shown on your statement", "Card was charged for something you did not purchase with the card", 0),
    (CC, "Fees or interest", None, 1),
    (CS, "Managing an account", "Deposits and withdrawals", 1),
    (CS, "Managing an account", None, 0),
    (CS, "Managing an account", "Deposits and withdrawals", 0),
    (CS, "Managing an account", "Deposits and withdrawals", 1),
]


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "complaints.sqlite"
    make_db(str(p), ROWS)
    return str(p)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_values_counts_and_blank_sub_issue(db, tmp_path):
    out = tmp_path / "values.csv"
    lines = []
    export.export_values(db, str(out), out=lines.append)
    rows = read_csv(out)
    assert rows[0] == ["product", "issue", "sub_issue", "complaints", "narratives"]
    body = rows[1:]
    assert body[0] == [CS, "Managing an account", "Deposits and withdrawals", "3", "2"]
    assert [CS, "Managing an account", "", "1", "0"] in body
    assert [CC, "Fees or interest", "", "1", "1"] in body
    assert sum(int(r[3]) for r in body) == len(ROWS)
    assert [r[0] for r in body] == sorted(r[0] for r in body)
    assert "4 product-issue-sub-issue rows, 7 complaints" in lines[0]


def test_table_csv_round_trips_nulls_commas_and_newlines(db, tmp_path):
    m = export.export_table(db, str(tmp_path / "out"), out=lambda *_: None)
    rows = read_csv(tmp_path / "out" / "complaints.csv")
    assert rows[0] == build_db.DB_COLUMNS
    assert len(rows) == 1 + len(ROWS) == 1 + m["rows"]
    first = dict(zip(rows[0], rows[1]))
    assert first["company"] == "BANK A, N.A."
    assert first["company_public_response"] == "Line one\nline \"two\""
    assert first["sub_product"] == "" and first["state"] == ""
    assert m["db_built_at"] == "2026-11-02T20:50:00Z"
    with open(tmp_path / "out" / "complaints.manifest.json", encoding="utf-8") as f:
        assert json.load(f)["sha256"] == export.sha256_file(tmp_path / "out" / "complaints.csv")


def test_table_refuses_a_file_over_the_console_limit(db, tmp_path):
    with pytest.raises(export.ExportError, match="upload limit"):
        export.export_table(db, str(tmp_path / "out"), limit=100, out=lambda *_: None)


def test_refuses_changed_columns(tmp_path):
    p = tmp_path / "odd.sqlite"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE complaints (complaint_id INTEGER, product TEXT)")
    conn.close()
    with pytest.raises(export.ExportError, match="columns differ"):
        export.export_values(str(p), str(tmp_path / "v.csv"))


def test_missing_database(tmp_path):
    with pytest.raises(export.ExportError, match="not found"):
        export.export_values(str(tmp_path / "none.sqlite"), str(tmp_path / "v.csv"))


def test_schema_matches_the_table():
    with open("analytics/schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    assert [c["name"] for c in schema] == build_db.DB_COLUMNS
    assert {c["name"] for c in schema if c["type"] == "DATE"} == {"date_received",
                                                                  "date_sent_to_company"}
