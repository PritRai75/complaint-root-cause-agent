"""Sampling and first-look tests on a small fake database, plus one check on the real data."""
import json
import os
import sqlite3

import pytest

from eval import taxonomy_sample as ts
from ingest import build_db, sources
from ingest.textnorm import fingerprint, normalize
from scripts import explore_data

CARD, CHK = sources.PRODUCTS

# (product, month, narratives in that month); every complaint also gets a
# no-narrative twin so the pool must come from the narratives table.
LAYOUT = [(CARD, "2026-01", 40), (CARD, "2026-02", 10), (CARD, "2026-03", 25),
          (CHK, "2026-01", 30), (CHK, "2026-02", 30), (CHK, "2026-08", 0)]
TEMPLATE = "I demand XXXX remove the fee of {$35.00} charged on XX/XX/XXXX."


def letters(n):
    """Spell a number in letters, so each text stays distinct after numbers are normalised."""
    return "".join("abcdefghij"[int(d)] for d in str(n))


def make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(build_db.SCHEMA)
    cid = 1000
    for product, month, n in LAYOUT:
        for i in range(n + 1):
            cid += 1
            has = i < n
            conn.execute(
                f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(build_db.DB_COLUMNS))})",
                (cid, f"{month}-1{i % 9}", product, "Sub", "Fees", None, "BANK A", "TX", None,
                 "Web", f"{month}-1{i % 9}",
                 "In progress" if i % 7 == 0 else "Closed with explanation",
                 None, None if i % 7 == 0 else "Yes", int(has), "A.zip"))
            if has:
                text = (TEMPLATE.replace("35.00", str(i)) if i % 5 == 0
                        else f"Complaint {letters(cid)} about <b>{product}</b> & more.")
                conn.execute("INSERT INTO narratives VALUES (?, ?, ?)", (cid, text, len(text)))
    conn.executemany("INSERT INTO build_info VALUES (?, ?)", [
        ("built_at", "2026-11-02T00:00:00Z"), ("loader_commit", "abc1234"),
        ("complaints_rows", "141"), ("narratives_rows", "135")])
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "fake.sqlite")
    make_db(path)
    return path


def test_allocate_is_proportional_and_exact():
    share = ts.allocate({"a": 40, "b": 10, "c": 25}, 30)
    assert share == {"a": 16, "b": 4, "c": 10}
    assert sum(ts.allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    assert ts.allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}  # ties by key
    assert ts.allocate({"a": 5, "b": 0}, 3) == {"a": 3, "b": 0}
    with pytest.raises(ts.SampleError):
        ts.allocate({"a": 2}, 3)


def test_draw_is_seeded_balanced_and_only_narratives(db):
    conn = ts.connect_ro(db)
    first = ts.create_sample(conn, per_product=20)
    again = ts.create_sample(conn, per_product=20)
    other = ts.create_sample(conn, per_product=20, seed=1)
    with_text = {r[0] for r in conn.execute("SELECT complaint_id FROM narratives")}
    conn.close()
    conn2 = ts.connect_ro(db)

    assert first["items"] == again["items"]
    assert first["items"] != other["items"]
    ids = [it["complaint_id"] for it in first["items"]]
    assert len(ids) == len(set(ids)) == 40
    assert set(ids) <= with_text
    texts = dict(conn2.execute("SELECT complaint_id, narrative FROM narratives"))
    conn2.close()
    fps = [fingerprint(texts[i]) for i in ids]
    assert len(set(fps)) == len(fps)  # one per distinct text
    template_ids = sorted(i for i, t in texts.items() if fingerprint(t) == fingerprint(TEMPLATE))
    assert not set(template_ids[1:]) & set(ids)  # only the lowest ID can stand for the template
    assert sum(it["product"] == CARD for it in first["items"]) == 20
    drawn = {(a["product"], a["month"]): a["drawn"] for a in first["allocation"]}
    assert not any(month == "2026-08" for _, month in drawn)  # no narratives, no stratum
    assert sum(drawn.values()) == 40
    assert [it["ref"] for it in first["items"]][:3] == ["R01", "R02", "R03"]
    assert first["database"]["loader_commit"] == "abc1234"


def test_run_saves_once_and_reuses(db, tmp_path, monkeypatch):
    sample_path, sheet_path = tmp_path / "s.json", tmp_path / "sheet" / "s.html"
    monkeypatch.setattr(ts, "create_sample",
                        lambda conn, f=ts.create_sample: f(conn, per_product=5))
    first = ts.run(db, str(sample_path), str(sheet_path))
    saved = json.loads(sample_path.read_text(encoding="utf-8"))
    assert saved["items"] == first["items"]

    saved["items"][0]["note"] = "kept"  # a reuse must read the file, not redraw
    sample_path.write_text(json.dumps(saved), encoding="utf-8")
    again = ts.run(db, str(sample_path), str(sheet_path))
    assert again["items"][0].get("note") == "kept"
    forced = ts.run(db, str(sample_path), str(sheet_path), force=True)
    assert "note" not in forced["items"][0]


def test_sheet_escapes_text_and_hides_company(db, tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "create_sample",
                        lambda conn, f=ts.create_sample: f(conn, per_product=30))
    ts.run(db, str(tmp_path / "s.json"), str(tmp_path / "s.html"))
    page = (tmp_path / "s.html").read_text(encoding="utf-8")
    assert "<b>Credit card</b>" not in page and "&lt;b&gt;" in page
    assert "BANK A" not in page
    assert page.count("<article") == 60
    assert "other narratives in scope" in page  # the template, if drawn, is flagged


def test_missing_ids_stop_the_sheet(db, tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"items": [{"ref": "R1", "complaint_id": 1,
                                           "product": CARD, "month": "2026-01"}]}),
                    encoding="utf-8")
    with pytest.raises(ts.SampleError, match="not in this database"):
        ts.run(db, str(path), str(tmp_path / "s.html"))


def test_fingerprint_ignores_redactions_amounts_and_case():
    a = "I was charged {$35.00} on XX/XX/XXXX by XXXX. Please refund!"
    b = "i was CHARGED {$1,200.50} on XX/XX/year> by XXXXXXXX please refund"
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(a + " Also my card was frozen.")
    assert "exxon" in normalize("My Exxon card")  # XX inside a word is not a redaction


def test_explore_report_runs(db, monkeypatch):
    conn = explore_data.connect_ro(db)
    text = "\n".join(explore_data.report(conn))
    conn.close()
    assert "== company_response by product" in text
    assert "Template duplicates" in text and "Largest 15 groups" in text
    assert "In progress" in text


# Real data: the committed sample must be exactly what the seeded draw gives.
@pytest.mark.skipif(not (os.path.exists(sources.DB_PATH) and os.path.exists(ts.SAMPLE_PATH)),
                    reason="database or committed sample not on this machine")
def test_committed_sample_reproduces():
    with open(ts.SAMPLE_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    conn = ts.connect_ro(sources.DB_PATH)
    fresh = ts.create_sample(conn, per_product=saved["per_product"], seed=saved["seed"])
    conn.close()
    assert fresh["items"] == saved["items"]
    assert fresh["allocation"] == saved["allocation"]
