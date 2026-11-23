"""Tests for analytics/recompute.py and analytics/spike_sample.py on fake data."""
import csv
import sqlite3

import pandas as pd
import pytest

from analytics import mapping
from analytics import recompute as rc
from analytics import spike_sample as ss
from ingest import build_db

CC, CS = "Credit card", "Checking or savings account"
FRAUD = (CC, "Problem with a purchase shown on your statement",
         "Card was charged for something you did not purchase with the card")
NEWACC = (CS, "Opening an account", "Account opened without my consent or knowledge")
OTHER = (CC, "Fees or interest", "")
FMAP = {
    FRAUD: {"fraud": "yes", "kind": "unauthorized_transaction", "borderline": "no", "reason": "x"},
    NEWACC: {"fraud": "yes", "kind": "new_account", "borderline": "no", "reason": "x"},
    OTHER: {"fraud": "no", "kind": "", "borderline": "no", "reason": "x"},
}


def frame(rows):
    """rows: (company, month, key, n, response, timely)."""
    out, cid = [], 1
    for company, month, key, n, resp, timely in rows:
        for _ in range(n):
            out.append({"complaint_id": cid, "date_received": f"2026-{month:02d}-10",
                        "product": key[0], "issue": key[1], "sub_issue": key[2] or None,
                        "company": company, "company_response": resp,
                        "timely_response": timely})
            cid += 1
    return pd.DataFrame(out)


def spike_rows(company, fraud_by_month, filler=40):
    rows = []
    for m, k in enumerate(fraud_by_month, 1):
        if k:
            rows.append((company, m, FRAUD, k, "Closed with explanation", "Yes"))
        rows.append((company, m, OTHER, filler, "Closed with explanation", "Yes"))
    return rows


def test_flag_joins_blank_sub_issue_and_refuses_unmapped():
    f = rc.flag(frame([("A", 1, OTHER, 2, "Closed", "Yes"), ("A", 1, FRAUD, 1, "Closed", "Yes")]),
                FMAP)
    assert f["is_fraud"].tolist() == [False, False, True]
    assert f["month"].iloc[0] == "2026-01-01"
    with pytest.raises(rc.RecomputeError, match="no mapping row"):
        rc.flag(frame([("A", 1, (CC, "New", ""), 1, "Closed", "Yes")]), FMAP)


def test_q1_counts_share_and_change():
    f = rc.flag(frame([("A", 1, FRAUD, 2, "C", "Yes"), ("A", 1, OTHER, 8, "C", "Yes"),
                       ("A", 2, FRAUD, 5, "C", "Yes"), ("B", 2, NEWACC, 1, "C", "Yes")]), FMAP)
    g = rc.q1(f).set_index(["product", "month"])
    jan, feb = g.loc[(CC, "2026-01-01")], g.loc[(CC, "2026-02-01")]
    assert (jan["complaints"], jan["fraud"], jan["fraud_share"]) == (10, 2, 0.2)
    assert pd.isna(jan["fraud_change"]) and feb["fraud_change"] == 3
    assert g.loc[(CS, "2026-02-01")]["new_account"] == 1


def test_q2_threshold_rank_and_wilson():
    f = rc.flag(frame([("A", 1, FRAUD, 50, "C", "Yes"), ("A", 1, OTHER, 150, "C", "Yes"),
                       ("B", 1, FRAUD, 20, "C", "Yes"), ("B", 1, OTHER, 180, "C", "Yes"),
                       ("C", 1, FRAUD, 199, "C", "Yes")]), FMAP)
    g = rc.q2(f).set_index("company")
    assert list(g.index) == ["A", "B"]          # C has 199 complaints
    assert (g.loc["A", "rate_rank"], g.loc["B", "rate_rank"]) == (1, 2)
    assert g.loc["A", "fraud_rate"] == 0.25
    assert g.loc["A", "wilson_low"] == pytest.approx(0.19508, abs=1e-5)   # by hand: 0.2547 - 0.0596


def test_q3_spike_rule_floors_and_biggest():
    rows = (spike_rows("A", [10, 10, 10, 40, 10, 10, 10, 10])      # April: spike, excess 30
            + spike_rows("B", [30, 30, 30, 25, 30, 30, 30, 30])    # never a spike
            + spike_rows("C", [1, 1, 1, 15, 1, 1, 1, 1])           # above threshold, under 20
            + spike_rows("D", [5, 5, 5, 5, 5, 5, 5, 90]))          # August: spike, not biggest
    g = rc.q3(rc.flag(frame(rows), FMAP)).set_index(["company", "month"])
    assert len(g) == 4 * 5                                          # April to August only
    a = g.loc[("A", "2026-04-01")]
    assert a["mean_prev3"] == 10 and a["sd_used"] == pytest.approx(10 ** 0.5)
    assert a["is_spike"] and a["is_biggest"] and a["excess"] == 30
    assert not g.loc[("C", "2026-04-01")]["is_spike"]
    assert g.loc[("D", "2026-08-01")]["is_spike"] and not g.loc[("D", "2026-08-01")]["is_biggest"]
    assert g["is_biggest"].sum() == 1 and g["is_spike"].sum() == 2


def test_q3_zero_fills_missing_months():
    rows = spike_rows("A", [0, 0, 0, 25, 0, 0, 0, 0])
    g = rc.q3(rc.flag(frame(rows), FMAP)).set_index(["company", "month"])
    apr = g.loc[("A", "2026-04-01")]
    assert apr["mean_prev3"] == 0 and apr["threshold"] == 0 and apr["is_spike"]


def test_q4_window_and_definitions():
    f = rc.flag(frame([
        ("A", 1, FRAUD, 4, "Closed with monetary relief", "Yes"),
        ("A", 2, FRAUD, 2, "Closed with non-monetary relief", "No"),
        ("A", 3, FRAUD, 3, "In progress", "Yes"),
        ("A", 4, FRAUD, 1, "Untimely response", "No"),
        ("A", 4, FRAUD, 1, None, None),
        ("A", 7, FRAUD, 9, "Closed with monetary relief", "Yes")]), FMAP)
    r = rc.q4(f).set_index(["product", "is_fraud"]).loc[(CC, True)]
    assert r["complaints"] == 11                       # July left out
    assert r["final_responses"] == 7                   # In progress and NULL left out
    assert r["timely"] == 4 and r["closed"] == 6
    assert (r["relief"], r["monetary_relief"]) == (6, 4)
    assert r["timely_rate"] == pytest.approx(4 / 7)


def write_bq_csv(path, df, floats):
    """Write a frame the way the BigQuery console saves results."""
    d = df.copy()
    for c, n in floats.items():
        d[c] = d[c].round(n)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(d.columns)
        for row in d.itertuples(index=False):
            w.writerow(["" if pd.isna(v) else str(v).lower() if isinstance(v, bool)
                        else int(v) if isinstance(v, float) and v == int(v) else v
                        for v in row])


def test_compare_accepts_rounding_and_finds_differences(tmp_path):
    f = rc.flag(frame([("A", 1, FRAUD, 3, "C", "Yes"), ("A", 1, OTHER, 6, "C", "Yes"),
                       ("A", 2, FRAUD, 1, "C", "Yes")]), FMAP)
    mine = rc.q1(f)
    cols = ["month", "product", "complaints", "fraud", "unauthorized_transaction",
            "new_account", "fraud_share", "fraud_change"]
    p = tmp_path / "q1.csv"
    write_bq_csv(p, mine[cols], {"fraud_share": 4})
    assert rc.compare("q1", mine, p) == []
    bad = mine[cols].copy()
    bad.loc[bad.index[0], "fraud"] = 4
    write_bq_csv(p, bad, {"fraud_share": 4})
    assert any("fraud: file 4" in d for d in rc.compare("q1", mine, p))


def make_db(path, df, narratives):
    conn = sqlite3.connect(path)
    conn.executescript(build_db.SCHEMA)
    for r in df.itertuples(index=False):
        conn.execute("INSERT INTO complaints (complaint_id, date_received, product, issue, "
                     "sub_issue, company, company_response, timely_response, has_narrative, "
                     "source_file) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (r.complaint_id, r.date_received, r.product, r.issue, r.sub_issue,
                      r.company, r.company_response, r.timely_response,
                      int(r.complaint_id in narratives), "x.zip"))
    for cid in narratives:
        conn.execute("INSERT INTO narratives VALUES (?, ?, ?)", (cid, f"story {cid}", 7))
    conn.commit()
    conn.close()


def test_recompute_end_to_end_and_missing_file(tmp_path):
    rows = spike_rows("A", [10, 10, 10, 40, 10, 10, 10, 10], filler=40)
    df = frame(rows)
    db = tmp_path / "c.sqlite"
    make_db(str(db), df, set())
    mp = tmp_path / "m.csv"
    with open(mp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(mapping.COLUMNS)
        for k, r in FMAP.items():
            w.writerow(list(k) + [r["fraud"], r["kind"], r["borderline"], r["reason"]])
    res = tmp_path / "results"
    res.mkdir()
    f = rc.flag(rc.load_complaints(str(db)), mapping.load(str(mp)))
    frames = {"q0": rc.q0(f, len(df)), "q1": rc.q1(f), "q2": rc.q2(f), "q2b": rc.q2b(f),
              "q3": rc.q3(f), "q4": rc.q4(f)}
    for name, (fname, _, floats) in rc.RESULTS.items():
        if name != "q4":
            write_bq_csv(res / fname, frames[name], floats)
    lines = []
    bad, _ = rc.recompute(str(db), str(res), str(mp), out=lines.append)
    assert bad == 1 and any(line.startswith("q4: MISSING") for line in lines)
    assert "q3: matches (5 rows)" in lines


def test_spike_sample_draws_seeded_fraud_narratives(tmp_path, monkeypatch):
    rows = spike_rows("A", [10, 10, 10, 40, 10, 10, 10, 10], filler=2)
    df = frame(rows)
    april_fraud = df[(df["date_received"] == "2026-04-10") & (df["issue"] == FRAUD[1])]
    april_other = df[(df["date_received"] == "2026-04-10") & (df["issue"] == OTHER[1])]
    narr = set(april_fraud["complaint_id"][:30]) | set(april_other["complaint_id"])
    db = tmp_path / "c.sqlite"
    make_db(str(db), df, narr)
    pool = ss.candidates("A", "2026-04-01", FMAP, str(db))
    assert len(pool) == 30                          # fraud only, with a narrative
    picked = ss.draw(pool)
    assert len(picked) == 25 and picked == ss.draw(pool)
    assert [r[0] for r in picked] == sorted(r[0] for r in picked)
    assert ss.draw(pool[:10]) == pool[:10]
    spikes = tmp_path / "q3.csv"
    spikes.write_text("company,month,is_biggest\nA,2026-04-01,true\nA,2026-08-01,false\n")
    assert ss.biggest(str(spikes)) == ("A", "2026-04-01")
    spikes.write_text("company,month,is_biggest\nA,2026-08-01,true\n")
    with pytest.raises(ss.SampleError, match="after narratives stop"):
        ss.biggest(str(spikes))
