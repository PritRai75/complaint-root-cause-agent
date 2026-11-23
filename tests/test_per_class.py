"""Tests for eval/per_class.py: per-code precision and recall, the untagged rule,
macro and micro F1, the support threshold, the fraud_claim spotlight, the
confusion matrix image, the refusals, and the real committed test log."""
import json
import os

import pytest

from eval import per_class as pc
from eval import tagrun
from eval import taxonomy

CODES = [c["code"] for c in taxonomy.load(None)["cause"]]
CC, CS = "Credit card", "Checking or savings account"
REAL_LOG = "eval/runs/test-20261116T055317Z.jsonl"

# ref, complaint_id, product, label, tag, status
ITEMS = [
    ("T001", 101, CC, "fraud_claim", "fraud_claim", "tagged"),
    ("T002", 102, CS, "fraud_claim", "fraud_claim", "tagged"),
    ("T003", 103, CS, "third_party", "fraud_claim", "tagged"),
    ("T004", 104, CS, "servicing_error", "account_restricted", "tagged"),
    ("T005", 105, CS, "account_restricted", "account_restricted", "tagged"),
    ("T006", 106, CC, "servicing_error", None, "failed"),
    ("T007", 107, CC, "dispute_mishandled", "dispute_mishandled", "tagged"),
    ("T008", 108, CC, "fee_charged", "terms_not_honored", "tagged"),
]


def item_rows(items=ITEMS, kind="test"):
    return [{"type": "item", "ref": ref, "set": kind, "complaint_id": cid, "product": product,
             "label": label, "uncertain": False, "tag": tag, "status": status}
            for ref, cid, product, label, tag, status in items]


def write_labels(d, items=ITEMS, name="test"):
    lines = [f'set = "{name}"', 'taxonomy_version = "v1"', ""]
    for ref, cid, product, label, *_ in items:
        lines += ["[[item]]", f'ref = "{ref}"', f"complaint_id = {cid}",
                  f'product = "{product}"', f'code = "{label}"', "uncertain = false",
                  'note = ""', ""]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_log(path, items=ITEMS, kind="test", final=True):
    rows = item_rows(items, kind)
    header = {"type": "header", "run_id": f"{kind}-20261116T000000Z", "set": kind,
              "final": final, "limit": None, "items": len(rows), "code_commit": "abc1234",
              "prompt_version": "tag-1", "taxonomy_version": "v1"}
    with open(path, "w", encoding="utf-8") as f:
        for r in [header] + rows + [{"type": "summary", **tagrun.score(rows)}]:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def setup(tmp_path):
    labels = tmp_path / "labels"
    write_labels(str(labels))
    log = tmp_path / "test-20261116T000000Z.jsonl"
    write_log(str(log))
    return {"labels": str(labels), "log": str(log), "runs": str(tmp_path / "runs"),
            "img": str(tmp_path / "img")}


def run(s, log=None):
    lines = []
    path, result = pc.analyse(log or s["log"], labels_dir=s["labels"], runs_dir=s["runs"],
                              img_dir=s["img"], out=lines.append)
    return path, result, "\n".join(lines)


def row(result, code):
    return next(r for r in result["rows"] if r["code"] == code)


def test_precision_and_recall_per_code():
    res = pc.per_class(item_rows(), CODES)
    fraud = row(res, "fraud_claim")
    assert (fraud["labelled"], fraud["tagged"], fraud["right"]) == (2, 3, 2)
    assert fraud["precision"]["value"] == pytest.approx(0.6667)
    assert fraud["recall"]["value"] == 1.0
    assert fraud["f1"] == pytest.approx(0.8)
    assert fraud["count_bias"] == 1.5
    acct = row(res, "account_restricted")
    assert (acct["precision"]["k"], acct["precision"]["n"]) == (1, 2)
    assert acct["f1"] == pytest.approx(0.6667)


def test_untagged_item_counts_against_recall_only():
    res = pc.per_class(item_rows(), CODES)
    serv = row(res, "servicing_error")
    assert (serv["recall"]["k"], serv["recall"]["n"]) == (0, 2)
    assert serv["precision"]["n"] == 0 and serv["precision"]["value"] is None
    assert serv["f1"] == 0.0 and serv["count_bias"] == 0.0
    assert res["matrix"]["servicing_error"][pc.UNTAGGED] == 1
    assert res["tagged"] == 7 and res["scored"] == 8


def test_code_never_labelled_has_no_count_bias():
    terms = row(pc.per_class(item_rows(), CODES), "terms_not_honored")
    assert (terms["labelled"], terms["tagged"]) == (0, 1)
    assert terms["count_bias"] is None and terms["f1"] == 0.0


def test_macro_over_all_ten_codes_and_micro():
    res = pc.per_class(item_rows(), CODES)
    assert res["macro_f1_codes"] == 10
    assert res["macro_f1"] == pytest.approx((0.8 + 1.0 + 2 / 3) / 10, abs=1e-4)
    assert res["right"] == 4
    assert (res["micro"]["precision"]["k"], res["micro"]["precision"]["n"]) == (4, 7)
    assert (res["micro"]["recall"]["k"], res["micro"]["recall"]["n"]) == (4, 8)
    assert res["micro"]["f1"] == pytest.approx(8 / 15, abs=1e-4)


def test_micro_f1_equals_exact_agreement_when_all_tagged():
    items = [it for it in ITEMS if it[5] == "tagged"]
    res = pc.per_class(item_rows(items), CODES)
    assert res["micro"]["f1"] == pytest.approx(res["right"] / res["scored"], abs=1e-4)


def test_intervals_only_from_ten():
    assert "wilson95" not in pc.rate(9, 9) and pc.rate(9, 9)["low_support"]
    r = pc.rate(9, 10)
    assert not r["low_support"]
    lo, hi = tagrun.wilson(9, 10)
    assert r["wilson95"] == [round(lo, 3), round(hi, 3)]
    assert pc.rate(0, 0)["value"] is None


def test_spotlight_lists_false_positives_and_misses():
    sp = pc.spotlight(pc.per_class(item_rows(), CODES), "fraud_claim")
    assert sp["false_positives_by_label"] == {"third_party": 1}
    assert sp["misses_by_tag"] == {}


def test_unknown_code_refused():
    items = item_rows() + item_rows([("T009", 109, CC, "fraud_claim", "fees", "tagged")])
    with pytest.raises(pc.PerClassError, match="not a taxonomy code"):
        pc.per_class(items, CODES)


def test_analyse_writes_json_and_the_same_image_twice(setup):
    path, r, text = run(setup)
    assert os.path.basename(path) == "per-class-test-20261116T000000Z.json"
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["right"] == 4 and saved["exact_agreement"]["k"] == 4
    assert saved["min_support"] == 10
    assert saved["spotlight"]["false_positives_by_label"] == {"third_party": 1}
    img = os.path.join(setup["img"], "confusion-test-20261116T000000Z.png")
    with open(img, "rb") as f:
        first = f.read()
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    run(setup)
    with open(img, "rb") as f:
        assert f.read() == first
    assert "Macro-F1 over 10 codes: 24.7%" in text
    assert "low support" in text
    assert "tagged fraud_claim but labelled: third_party 1" in text


def test_refuses_a_test_log_not_from_final(setup, tmp_path):
    log = tmp_path / "not-final.jsonl"
    write_log(str(log), final=False)
    with pytest.raises(Exception, match="--final"):
        run(setup, str(log))


def test_refuses_changed_labels(setup):
    changed = list(ITEMS)
    changed[0] = ("T001", 101, CC, "dispute_mishandled", "fraud_claim", "tagged")
    write_labels(setup["labels"], changed)
    with pytest.raises(Exception, match="label file has changed"):
        run(setup)


def test_real_test_log(tmp_path):
    """The committed tagger test run: needs no database, so it runs in CI too."""
    _, r = pc.analyse(REAL_LOG, runs_dir=str(tmp_path / "runs"),
                         img_dir=str(tmp_path / "img"), out=lambda *_: None)
    assert (r["right"], r["scored"], r["tagged"]) == (163, 200, 200)
    assert r["micro"]["f1"] == pytest.approx(0.815, abs=1e-4)
    fraud = row(r, "fraud_claim")
    assert (fraud["precision"]["k"], fraud["precision"]["n"]) == (55, 62)
    assert (fraud["recall"]["k"], fraud["recall"]["n"]) == (55, 55)
    assert row(r, "third_party")["recall"]["k"] == 1
    assert row(r, "third_party")["precision"]["low_support"]
    assert r["spotlight"]["false_positives_by_label"]["third_party"] == 3
    assert sum(r["spotlight"]["false_positives_by_label"].values()) == 7
