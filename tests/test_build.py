"""Loader tests on small fake archives. They need no real data, so CI runs them."""
import csv
import io
import json
import sqlite3
import zipfile

import pytest

from ingest import build_db, fetch

HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company",
    "State", "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]
PRODUCTS = ("Credit card", "Checking or savings account")


def row(cid, date, product, narrative="", **extra):
    base = {
        "Date received": date, "Product": product, "Sub-product": "General-purpose credit card",
        "Issue": "Fees or interest", "Sub-issue": "", "Consumer complaint narrative": narrative,
        "Company public response": "", "Company": "BANK A", "State": "TX", "ZIP code": "750XX",
        "Tags": "", "Submitted via": "Web", "Date sent to company": date,
        "Company response to consumer": "Closed with explanation", "Timely response?": "Yes",
        "Complaint ID": str(cid),
    }
    base.update(extra)
    return base


def make_zip(path, rows, header=HEADER):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.stem + ".csv", buf.getvalue())


TRICKY = "I was charged {$35.00} on XX/XX/XXXX by XXXX,\nand \"nobody\" called back."


@pytest.fixture
def archive(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_zip(raw / "A.zip", [
        row(1, "2026-01-05", "Credit card", TRICKY),
        row(2, "2026-01-06", "Checking or savings account"),
        row(3, "2026-01-07", "Credit reporting or other personal consumer reports", "out of scope"),
        row(4, "2025-12-31", "Credit card", "before the window"),
        row(5, "2026-02-01", "Credit card", "   \n  "),
    ])
    make_zip(raw / "B.zip", [
        row(6, "2026-08-31", "Checking or savings account", "Overdraft fee again."),
        row(7, "2026-09-01", "Credit card", "after the window"),
    ])
    files = ["A.zip", "B.zip"]
    manifest = tmp_path / "manifest.json"
    fetch.fetch(files=files, raw_dir=str(raw), manifest_path=str(manifest),
                download=lambda url, dest: pytest.fail("should not download"))
    return {"raw": str(raw), "manifest": str(manifest), "files": files,
            "db": str(tmp_path / "out.sqlite")}


def run_build(a, **kw):
    return build_db.build(files=a["files"], raw_dir=a["raw"], manifest_path=a["manifest"],
                          db_path=a["db"], products=PRODUCTS,
                          date_from="2026-01-01", date_to="2026-08-31", **kw)


def test_filters_and_tables(archive):
    run_build(archive)
    conn = sqlite3.connect(archive["db"])
    ids = [r[0] for r in conn.execute("SELECT complaint_id FROM complaints ORDER BY 1")]
    assert ids == [1, 2, 5, 6]
    narr = dict(conn.execute("SELECT complaint_id, narrative FROM narratives"))
    assert set(narr) == {1, 6}
    assert narr[1] == TRICKY  # stored exactly as published
    flags = dict(conn.execute("SELECT complaint_id, has_narrative FROM complaints"))
    assert flags == {1: 1, 2: 0, 5: 0, 6: 1}  # whitespace-only counts as no narrative
    cols = [r[1] for r in conn.execute("PRAGMA table_info(complaints)")]
    assert "zip_code" not in cols and "narrative" not in cols
    assert conn.execute("SELECT sub_issue FROM complaints WHERE complaint_id = 1").fetchone()[0] is None
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    assert info["complaints_rows"] == "4" and info["narratives_rows"] == "2"
    sources = json.loads(info["source_files"])
    assert [s["rows_scanned"] for s in sources] == [5, 2]
    conn.close()


def test_refuses_to_overwrite_without_force(archive):
    run_build(archive)
    with pytest.raises(build_db.BuildError, match="--force"):
        run_build(archive)
    run_build(archive, force=True)


def test_changed_file_is_rejected(archive):
    with open(f"{archive['raw']}/A.zip", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(build_db.BuildError, match="SHA-256"):
        run_build(archive)
    with pytest.raises(SystemExit):
        fetch.fetch(files=archive["files"], raw_dir=archive["raw"],
                    manifest_path=archive["manifest"])


def test_new_column_stops_build(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_zip(raw / "A.zip", [row(1, "2026-01-05", "Credit card")], header=HEADER + ["Surprise"])
    manifest = tmp_path / "m.json"
    fetch.fetch(files=["A.zip"], raw_dir=str(raw), manifest_path=str(manifest))
    with pytest.raises(build_db.BuildError, match="Surprise"):
        build_db.build(files=["A.zip"], raw_dir=str(raw), manifest_path=str(manifest),
                       db_path=str(tmp_path / "x.sqlite"))
    assert not (tmp_path / "x.sqlite").exists()
    assert not (tmp_path / "x.sqlite.building").exists()


def test_bad_date_stops_build():
    bad = row(1, "08/01/2026", "Credit card")
    with pytest.raises(build_db.BuildError, match="YYYY-MM-DD"):
        build_db.to_records(bad, "A.zip", PRODUCTS, "2026-01-01", "2026-08-31")
