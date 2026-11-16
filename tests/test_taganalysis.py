"""Tests for eval/taganalysis.py: runner-up matches, the issue table and its bound,
and the refusals. Fake labels, fake run logs and a fake complaints database."""
import json
import os
import sqlite3

import pytest

from eval import taganalysis as ta
from eval import tagrun

CODES = {"fraud_claim", "dispute_mishandled", "servicing_error", "account_restricted",
         "terms_not_honored", "fee_charged", "credit_reporting", "third_party", "other",
         "unclear"}
CC, CS = "Credit card", "Checking or savings account"

# ref, complaint_id, product, label, note, tag, status, issue
ITEMS = [
    ("T001", 101, CC, "fraud_claim", "vs dispute_mishandled. x", "dispute_mishandled", "tagged",
     "Problem with a purchase shown on your statement"),
    ("T002", 102, CC, "dispute_mishandled", "", "dispute_mishandled", "tagged",
     "Problem with a purchase shown on your statement"),
    ("T003", 103, CC, "fee_charged", "vs servicing_error. y", "terms_not_honored", "tagged",
     "Fees or interest"),
    ("T004", 104, CS, "account_restricted", "vs fraud_claim. z", None, "failed",
     "Closing an account"),
    ("T005", 105, CS, "account_restricted", "", "account_restricted", "tagged",
     "Closing an account"),
    ("T006", 106, CS, "fraud_claim", "vs third_party", "third_party", "tagged",
     "Problem using a debit or ATM card"),
    ("T007", 107, CC, "dispute_mishandled", "", "dispute_mishandled", "tagged",
     "Problem with a purchase shown on your statement"),
    ("T008", 108, CC, "fraud_claim", "", "fraud_claim", "tagged",
     "Problem with a purchase shown on your statement"),
]


def write_labels(d, items=ITEMS, name="test"):
    lines = [f'set = "{name}"', 'taxonomy_version = "v1"', ""]
    for ref, cid, product, label, note, *_ in items:
        lines += ["[[item]]", f'ref = "{ref}"', f"complaint_id = {cid}",
                  f'product = "{product}"', f'code = "{label}"', "uncertain = false",
                  f"note = {json.dumps(note)}", ""]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def log_rows(items=ITEMS, kind="test", final=True, limit=None):
    rows = [{"type": "item", "ref": ref, "set": kind, "complaint_id": cid, "product": product,
             "label": label, "uncertain": False, "tag": tag, "status": status}
            for ref, cid, product, label, note, tag, status, _ in items]
    header = {"type": "header", "run_id": f"{kind}-20261116T000000Z", "set": kind,
              "final": final, "limit": limit, "items": len(rows), "code_commit": "abc1234",
              "prompt_version": "tag-1", "taxonomy_version": "v1"}
    return [header] + rows + [{"type": "summary", **tagrun.score(rows)}]


def write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_db(path, items=ITEMS, product_of=None):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE complaints (complaint_id INTEGER PRIMARY KEY, product TEXT, "
                 "issue TEXT, sub_issue TEXT)")
    for ref, cid, product, label, note, tag, status, issue in items:
        conn.execute("INSERT INTO complaints VALUES (?, ?, ?, ?)",
                     (cid, (product_of or {}).get(cid, product), issue, None))
    conn.commit()
    conn.close()


@pytest.fixture
def setup(tmp_path):
    labels = tmp_path / "labels"
    write_labels(str(labels))
    log = tmp_path / "test-20261116T000000Z.jsonl"
    write_log(str(log), log_rows())
    db = tmp_path / "complaints.sqlite"
    write_db(str(db))
    return {"labels": str(labels), "log": str(log), "db": str(db),
            "runs": str(tmp_path / "runs")}


def run(s, **kw):
    lines = []
    args = dict(labels_dir=s["labels"], db_path=s["db"], runs_dir=s["runs"], out=lines.append)
    args.update(kw)
    path, result = ta.analyse(s["log"], **args)
    return path, result, "\n".join(lines)


def test_runner_up_reads_only_a_leading_vs_with_a_known_code():
    assert ta.runner_up("vs third_party. The request is aimed at the merchant", CODES) == "third_party"
    assert ta.runner_up("vs fee_charged", CODES) == "fee_charged"
    assert ta.runner_up("vs fees. typo", CODES) is None
    assert ta.runner_up("The consumer vs the bank: vs fraud_claim", CODES) is None
    assert ta.runner_up("", CODES) is None
    assert ta.runner_up(None, CODES) is None


def test_real_test_labels_all_parse():
    notes = ta.load_notes("test")
    with_vs = [n for _, _, n in notes.values() if n.startswith("vs ")]
    assert len(notes) == 200
    assert len(with_vs) == 81
    assert all(ta.runner_up(n, CODES) for n in with_vs)


def test_misses_and_runner_up_matches(setup):
    _, r, text = run(setup)
    ru = r["runner_up"]
    # misses: T001 (match), T003 (no: tag terms, runner-up servicing), T004 (untagged: never
    # a match), T006 (match)
    assert ru["misses"] == 4
    assert ru["with_runner_up_note"] == 4
    assert ru["match_runner_up"] == 2
    assert ru["by_product"] == {CS: {"misses": 2, "match_runner_up": 1},
                                CC: {"misses": 2, "match_runner_up": 1}}
    assert ru["by_label"]["account_restricted"] == {"misses": 1, "match_runner_up": 0}
    assert ru["by_label"]["fraud_claim"] == {"misses": 2, "match_runner_up": 2}
    t004 = next(i for i in ru["items"] if i["ref"] == "T004")
    assert t004["tag"] == "failed" and t004["match"] is False
    assert "tag equals the noted runner-up: 2 of 4" in text


def test_issue_table_bound_and_tagger_on_the_same_items(setup):
    _, r, text = run(setup)
    t = r["issue_column"]
    assert t["scored"] == 8
    assert t["groups"] == 4
    big = t["rows"][0]
    assert (big["product"], big["issue"], big["n"]) == (CC, "Problem with a purchase shown on your statement", 4)
    # labels there: fraud 2, dispute 2: a tie, so the majority is 2 whichever is picked
    assert big["causes"] == 2 and big["majority"] == 2
    assert big["majority_labels"] == ["dispute_mishandled", "fraud_claim"]
    assert big["tagger_agree"] == 3
    # bound: 2 (purchase) + 1 (fees) + 2 (closing) + 1 (debit card) = 6 of 8
    assert t["issue_only_bound"]["k"] == 6
    assert t["tagger_exact"]["k"] == r["exact_agreement"]["k"] == 4
    assert t["groups_with_2plus_causes"] == 1 and t["items_in_those_groups"] == 4
    assert "Issue-only bound" in text


def test_same_issue_name_under_two_products_is_two_groups(tmp_path):
    items = [
        ("T001", 1, CC, "fee_charged", "", "fee_charged", "tagged", "Fees or interest"),
        ("T002", 2, CS, "servicing_error", "", "servicing_error", "tagged", "Fees or interest"),
    ]
    write_labels(str(tmp_path / "l"), items)
    write_log(str(tmp_path / "log.jsonl"), log_rows(items))
    write_db(str(tmp_path / "db.sqlite"), items)
    s = {"labels": str(tmp_path / "l"), "log": str(tmp_path / "log.jsonl"),
         "db": str(tmp_path / "db.sqlite"), "runs": str(tmp_path / "runs")}
    _, r, _ = run(s)
    assert r["issue_column"]["groups"] == 2
    assert r["issue_column"]["issue_only_bound"]["k"] == 2


def test_writes_the_analysis_file(setup):
    path, r, _ = run(setup)
    assert os.path.basename(path) == "analysis-test-20261116T000000Z.json"
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["runner_up"]["match_runner_up"] == 2
    assert saved["log_sha256"] == ta.sha256_file(setup["log"])
    assert saved["final"] is True and saved["run_commit"] == "abc1234"


def test_refuses_a_test_log_that_is_not_final(setup):
    write_log(setup["log"], log_rows(final=False))
    with pytest.raises(ta.AnalysisError, match="--final"):
        run(setup)


def test_refuses_a_limited_run(setup):
    write_log(setup["log"], log_rows(limit=3))
    with pytest.raises(ta.AnalysisError, match="--limit"):
        run(setup)


def test_refuses_an_unfinished_log(setup):
    write_log(setup["log"], log_rows()[:-1])
    with pytest.raises(ta.AnalysisError, match="didn't finish"):
        run(setup)


def test_refuses_changed_labels(setup):
    changed = list(ITEMS)
    changed[1] = ("T002", 102, CC, "servicing_error", "", "dispute_mishandled", "tagged", "x")
    write_labels(setup["labels"], changed)
    with pytest.raises(ta.AnalysisError, match="T002: the label file has changed"):
        run(setup)


def test_refuses_a_summary_that_does_not_match_the_items(setup):
    rows = log_rows()
    rows[-1]["agree"] += 1
    write_log(setup["log"], rows)
    with pytest.raises(ta.AnalysisError, match="re-scoring"):
        run(setup)


def test_refuses_ids_missing_from_the_database(setup):
    os.remove(setup["db"])
    write_db(setup["db"], ITEMS[:-1])
    with pytest.raises(ta.AnalysisError, match="1 complaint IDs not in"):
        run(setup)


def test_refuses_a_product_mismatch(setup):
    os.remove(setup["db"])
    write_db(setup["db"], product_of={101: CS})
    with pytest.raises(ta.AnalysisError, match="T001: product differs"):
        run(setup)


def test_refuses_a_missing_database(setup):
    with pytest.raises(ta.AnalysisError, match="not found"):
        run(setup, db_path=setup["db"] + ".nope")


def test_main_prints_one_line_errors(setup, capsys):
    assert ta.main(["x", setup["log"] + ".missing"]) == 1
    assert capsys.readouterr().out.startswith("ERROR ")
    assert ta.main(["x"]) == 2


def test_wilson_is_clamped_to_0_and_1():
    lo, hi = tagrun.wilson(0, 96)
    assert lo == 0.0 and str(round(lo, 3)) == "0.0"
    lo, hi = tagrun.wilson(96, 96)
    assert hi == 1.0
