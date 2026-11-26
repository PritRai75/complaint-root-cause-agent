"""Tests for analytics/routing_size.py: the share and estimate arithmetic, the
pool check, the mapping lookup, and a full run on a small fake database."""
import csv
import json
import os
import sqlite3

import pytest

from analytics import routing_size as rs
from eval import labelset as ls
from eval.tagrun import wilson
from ingest import build_db, sources

CARD, CHK = sources.PRODUCTS


def item(cid, product, label, tag, status="tagged"):
    return {"type": "item", "set": "test", "complaint_id": cid, "product": product,
            "label": label, "tag": tag, "status": status}


def test_size_counts_shares_and_estimates():
    items = [
        item(1, CARD, "fraud_claim", "fraud_claim"),      # fraud issue
        item(2, CARD, "fee_charged", "fee_charged"),      # fraud issue
        item(3, CARD, "fraud_claim", "fraud_claim"),      # other issue, right
        item(4, CARD, "third_party", "fraud_claim"),      # other issue, wrong tag
        item(5, CARD, "fee_charged", "fee_charged"),      # other issue
        item(6, CARD, "fraud_claim", None, "failed"),     # other issue, untagged
        item(7, CHK, "fraud_claim", "fraud_claim"),       # fraud issue
        item(8, CHK, "fee_charged", "fee_charged"),       # other issue
    ]
    fraud = {1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: True, 8: False}
    rows = rs.size(items, fraud, {CARD: 700, CHK: 350}, {CARD: 7, CHK: 7},
                   {CARD: 1000, CHK: 500})
    card, chk, both = rows
    assert card["test_items"] == 6 and card["test_fraud_issue"] == 2
    assert card["test_fraud_issue_fraud_claim"] == 1
    assert card["test_other_issue"] == 4 and card["other_fraud_claim"] == 2
    assert card["share"] == 0.5
    lo, hi = wilson(2, 4)
    assert card["share_low"] == round(lo, 4) and card["share_high"] == round(hi, 4)
    assert card["other_tagged_fraud_claim"] == 2 and card["other_tagged_right"] == 1
    assert card["est_per_month"] == 50.0                  # 0.5 x 700 / 7
    assert card["est_low"] == round(lo * 100, 1) and card["est_high"] == round(hi * 100, 1)
    assert chk["other_fraud_claim"] == 0 and chk["est_per_month"] == 0.0
    assert both["product"] == rs.BOTH
    assert both["test_items"] == 8 and both["pool_other_issue"] == 1050
    assert both["est_per_month"] == 50.0
    assert both["est_high"] == round(card["est_high"] + chk["est_high"], 1)
    assert both["share"] == ""                            # no pooled share across products
    assert list(both) == rs.COLUMNS


def test_size_refuses_a_product_with_no_items():
    with pytest.raises(rs.SizeError, match="no test items"):
        rs.size([item(1, CARD, "fraud_claim", "fraud_claim")], {1: False},
                {CARD: 1, CHK: 1}, {CARD: 1, CHK: 1}, {CARD: 1, CHK: 1})


def test_check_pool_names_the_months_that_differ():
    saved = [{"product": CARD, "month": "2026-01", "pool": 10, "drawn": 2},
             {"product": CARD, "month": "2026-02", "pool": 8, "drawn": 2}]
    rs.check_pool({(CARD, "2026-01"): 10, (CARD, "2026-02"): 8}, saved)
    with pytest.raises(rs.SizeError, match="2026-02: saved 8, rebuilt 9"):
        rs.check_pool({(CARD, "2026-01"): 10, (CARD, "2026-02"): 9}, saved)


def test_is_fraud_matches_a_blank_sub_issue_and_refuses_unknown_values():
    fmap = {(CARD, "Getting a credit card", "Card opened without my consent or knowledge"):
            {"fraud": "yes"},
            (CARD, "Closing your account", ""): {"fraud": "no"}}
    assert rs.is_fraud(fmap, CARD, "Getting a credit card",
                       "Card opened without my consent or knowledge")
    assert not rs.is_fraud(fmap, CARD, "Closing your account", None)
    with pytest.raises(rs.SizeError, match="not in the fraud mapping"):
        rs.is_fraud(fmap, CARD, "Something new", None)


# A full run on a fake database: 2 products x 2 months x 20 narratives. In each
# product, every fourth complaint is under a fraud issue; every tenth is a
# copy of one template text, so the pool drops its copies.
FRAUD_ISSUE = ("Fraud issue", "Unauthorised")
OTHER_ISSUE = ("Fees", None)


def letters(n):
    """Digits as letters: the fingerprint ignores digits, so texts must differ in letters."""
    return "".join("abcdefghij"[int(d)] for d in str(n))


def make_fake(tmp_path):
    db = str(tmp_path / "fake.sqlite")
    conn = sqlite3.connect(db)
    conn.executescript(build_db.SCHEMA)
    cid = 9000
    for product in sources.PRODUCTS:
        for month in ("2026-01", "2026-02"):
            for i in range(20):
                cid += 1
                issue, sub = FRAUD_ISSUE if i % 4 == 0 else OTHER_ISSUE
                conn.execute(
                    f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(build_db.DB_COLUMNS))})",
                    (cid, f"{month}-1{i % 9}", product, "Sub", issue, sub, "BANK", "TX", None,
                     "Web", f"{month}-1{i % 9}", "Closed with explanation", None, "Yes", 1,
                     "A.zip"))
                text = ("The same template text." if i % 10 == 5
                        else f"Narrative {letters(cid)} about {product}.")
                conn.execute("INSERT INTO narratives VALUES (?, ?, ?)", (cid, text, len(text)))
    # One complaint with no narrative, which the context count must leave out.
    cid += 1
    conn.execute(
        f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(build_db.DB_COLUMNS))})",
        (cid, "2026-02-01", CARD, "Sub", "Fees", None, "BANK", "TX", None, "Web",
         "2026-02-01", "Closed with explanation", None, "Yes", 0, "A.zip"))
    conn.executemany("INSERT INTO build_info VALUES (?, ?)", [
        ("built_at", "2026-11-02T00:00:00Z"), ("loader_commit", "abc1234"),
        ("complaints_rows", str(cid - 9000)), ("narratives_rows", str(cid - 9001))])
    conn.commit()
    conn.close()

    fmap = tmp_path / "fraud_mapping.csv"
    with open(fmap, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["product", "issue", "sub_issue", "fraud", "kind", "borderline", "reason"])
        for product in sources.PRODUCTS:
            w.writerow([product, *FRAUD_ISSUE, "yes", "unauthorized_transaction", "no", "Fraud."])
            w.writerow([product, "Fees", "", "no", "", "no", "Not fraud."])
    return db, str(fmap)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    db, fmap = make_fake(tmp_path)
    sets = tmp_path / "sets"
    sets.mkdir()
    # An excluded set holding one ordinary narrative, so the pool leaves it out.
    reading = tmp_path / "taxonomy_sample.json"
    reading.write_text(json.dumps({"items": [
        {"ref": "R1", "complaint_id": 9002, "product": CARD, "month": "2026-01"}]}),
        encoding="utf-8")
    monkeypatch.setattr(ls.tx, "READING_PATH", str(reading))
    monkeypatch.setattr(ls, "SETS_DIR", str(sets))
    monkeypatch.setattr(ls, "DRAWS", {"test": ("T", 10, 5)})
    conn = ls.ts.connect_ro(db)
    test = ls.create_set(conn, "test", ["reading"], "v1")
    conn.close()
    set_path = tmp_path / "test.json"
    set_path.write_text(json.dumps(test), encoding="utf-8")
    # Labels: every item whose complaint ID is odd is a fraud claim; the tagger
    # tags every item fraud_claim, so its right tags equal the labels.
    log = tmp_path / "log.jsonl"
    with open(log, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "run_id": "fake"}) + "\n")
        for it in test["items"]:
            label = "fraud_claim" if it["complaint_id"] % 2 else "fee_charged"
            f.write(json.dumps(item(it["complaint_id"], it["product"], label,
                                    "fraud_claim")) + "\n")
    return {"db": db, "map": fmap, "set": str(set_path), "log": str(log), "test": test,
            "out": str(tmp_path / "results" / "q7.csv")}


def test_run_on_a_fake_database(fake):
    lines = []
    rows = rs.run(db_path=fake["db"], set_path=fake["set"], log_path=fake["log"],
                  mapping_path=fake["map"], out_path=fake["out"], out=lines.append)
    with open(fake["out"], encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert [r["product"] for r in saved] == [CARD, CHK, rs.BOTH]
    assert list(saved[0]) == rs.COLUMNS

    # Expected values worked out independently of routing_size.
    conn = sqlite3.connect(fake["db"])
    issue = {cid: (iss, sub) for cid, iss, sub in conn.execute(
        "SELECT complaint_id, issue, sub_issue FROM complaints")}
    conn.close()
    for r in rows[:2]:
        mine = [it for it in fake["test"]["items"] if it["product"] == r["product"]]
        other = [it for it in mine if issue[it["complaint_id"]] == OTHER_ISSUE]
        k = sum(it["complaint_id"] % 2 for it in other)
        assert r["test_items"] == 10
        assert r["test_other_issue"] == len(other)
        assert r["other_fraud_claim"] == k
        assert r["other_tagged_fraud_claim"] == len(other)
        assert r["other_tagged_right"] == k
        assert r["pool_months"] == 2
    # Pool per product, other issues: 40 complaints, 30 under Fees. Of those
    # 30, i % 10 == 5 (template) occurs at i = 5 and 15 in each month, 4 a
    # product; only the very first card copy (9006) stands for the text, so
    # card keeps 27 (30 - 4 + 1), minus 9002 from the excluded set = 26, and
    # checking keeps 26 (30 - 4).
    assert rows[0]["pool_other_issue"] == 26
    assert rows[1]["pool_other_issue"] == 26
    # Context: every narrative under Fees, copies included; the complaint with
    # no narrative is left out.
    assert rows[0]["narratives_other_issue"] == 30
    assert rows[1]["narratives_other_issue"] == 30
    assert rows[2]["est_per_month"] == round(rows[0]["est_per_month"] + rows[1]["est_per_month"], 1)
    assert lines[-1] == f"Written: {fake['out']}"


def test_run_refuses_a_log_from_another_set(fake, tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(item(1, CARD, "fraud_claim", "fraud_claim")) + "\n",
                     encoding="utf-8")
    with pytest.raises(rs.SizeError, match="not the test set's items"):
        rs.run(db_path=fake["db"], set_path=fake["set"], log_path=str(other),
               mapping_path=fake["map"], out_path=fake["out"], out=lambda s: None)


def test_run_refuses_a_pool_that_differs_from_the_saved_one(fake, tmp_path):
    test = dict(fake["test"])
    test["allocation"] = [dict(a) for a in test["allocation"]]
    test["allocation"][0]["pool"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(test), encoding="utf-8")
    with pytest.raises(rs.SizeError, match="rebuilt pool doesn't match"):
        rs.run(db_path=fake["db"], set_path=str(path), log_path=fake["log"],
               mapping_path=fake["map"], out_path=fake["out"], out=lambda s: None)


@pytest.mark.skipif(not os.path.exists(sources.DB_PATH), reason="database not on this machine")
def test_committed_result_reproduces(tmp_path):
    if not os.path.exists(rs.OUT_PATH):
        pytest.skip("q7_routing_size.csv not committed yet")
    out = str(tmp_path / "q7.csv")
    rs.run(out_path=out, out=lambda s: None)
    with open(out, encoding="utf-8") as a, open(rs.OUT_PATH, encoding="utf-8") as b:
        assert a.read() == b.read()
