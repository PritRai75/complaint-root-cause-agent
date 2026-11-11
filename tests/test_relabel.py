"""The blind re-label tool: draw, sheet, label file, check and score.

Draw tests use the committed test set; nothing here names which items are drawn."""
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone

import pytest

from eval import labelset as ls
from eval import relabel as rl
from ingest import build_db, sources

CARD, CHK = sources.PRODUCTS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 11, 11, 16, 39, 43, tzinfo=timezone.utc)


def committed_test_set():
    with open(os.path.join(ROOT, "eval/sets/test.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A working copy with the committed taxonomy, test set and test labels, and a
    database holding a narrative for every test item."""
    for rel in ("eval/taxonomy.toml", "eval/taxonomy_sample.json", "eval/sets/test.json",
                "eval/labels/test.toml"):
        os.makedirs(tmp_path / os.path.dirname(rel), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), tmp_path / rel)
    db = tmp_path / "complaints.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(build_db.SCHEMA)
    for it in committed_test_set()["items"]:
        text = f"Narrative of complaint {it['complaint_id']} <b>as published</b>."
        conn.execute("INSERT INTO narratives VALUES (?, ?, ?)",
                     (it["complaint_id"], text, len(text)))
    conn.commit()
    conn.close()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fill(repo, codes=None, edit=None):
    """Fill the re-label file: by default each item gets its first code."""
    saved = ls.load_set(rl.NAME)
    first = {it["ref"]: it for it in
             __import__("tomllib").load(open("eval/labels/test.toml", "rb"))["item"]}
    text = open(ls.label_path(rl.NAME), encoding="utf-8").read()
    for it in saved["items"]:
        code = (codes or {}).get(it["ref"], first[it["test_ref"]]["code"])
        text = text.replace(f'ref = "{it["ref"]}"\nproduct = {json.dumps(it["product"])}\n'
                            'code = ""', f'ref = "{it["ref"]}"\nproduct = '
                            f'{json.dumps(it["product"])}\ncode = "{code}"', 1)
    if edit:
        text = edit(text)
    open(ls.label_path(rl.NAME), "w", encoding="utf-8").write(text)
    return saved, first


# ---------- the draw ----------

def test_draw_is_15_per_product_seeded_and_leaves_out_discussed_items():
    test_set = committed_test_set()
    items = rl.draw(test_set)
    assert len(items) == 30
    assert sum(it["product"] == CARD for it in items) == 15
    assert [it["ref"] for it in items] == [f"B{i:02d}" for i in range(1, 31)]
    refs = [it["test_ref"] for it in items]
    assert len(set(refs)) == 30 and not set(refs) & set(rl.DISCUSSED)
    by_ref = {it["ref"]: it for it in test_set["items"]}
    assert all(by_ref[it["test_ref"]]["complaint_id"] == it["complaint_id"] and
               by_ref[it["test_ref"]]["product"] == it["product"] for it in items)
    assert rl.draw(test_set) == items
    assert refs != sorted(refs)                               # shuffled, not in T order


def test_discussed_items_are_real_test_items():
    refs = {it["ref"] for it in committed_test_set()["items"]}
    assert len(rl.DISCUSSED) == 14 and set(rl.DISCUSSED) <= refs


# ---------- prepare ----------

def test_prepare_records_the_gap_since_the_first_labels(repo):
    saved = rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    assert saved["gap_days_at_draw"] == 1.0 and saved["first_labels_commit"] == "e7efa08"
    assert ls.load_set(rl.NAME)["gap_days_at_draw"] == 1.0


def test_prepare_writes_a_blind_sheet_and_label_file(repo):
    saved = rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    assert os.path.exists("eval/sets/relabel.json")
    sheet = open(ls.sheet_path(rl.NAME), encoding="utf-8").read()
    labels = open(ls.label_path(rl.NAME), encoding="utf-8").read()
    assert sheet.count("<article") == 30 and "B01" in sheet
    assert "&lt;b&gt;as published&lt;/b&gt;" in sheet
    first = open("eval/labels/test.toml", encoding="utf-8").read()
    for it in saved["items"]:
        assert it["test_ref"] not in sheet and it["test_ref"] not in labels
        assert str(it["complaint_id"]) not in labels
    assert "complaint_id" not in labels and 'code = ""' in labels
    assert "vs third_party" in first and "vs third_party" not in sheet


def test_prepare_reuses_the_draw_and_never_overwrites_labels(repo):
    a = rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    fill(repo)
    before = open(ls.label_path(rl.NAME), encoding="utf-8").read()
    b = rl.prepare(db_path=str(repo / "complaints.sqlite"), now=datetime(2026, 11, 13, tzinfo=timezone.utc))
    assert a["items"] == b["items"]
    assert open(ls.label_path(rl.NAME), encoding="utf-8").read() == before


# ---------- check ----------

def test_check_shows_progress_and_problems(repo):
    rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    problems, lines = rl.check()
    assert problems == [] and lines[0].startswith("relabel: 0 of 30 coded, blank: B01")
    fill(repo, edit=lambda t: t.replace('code = "', 'code = "nope_', 1))
    problems, lines = rl.check()
    assert any("unknown code 'nope_" in p for p in problems)
    assert lines[0] == "relabel: 29 of 30 coded"


def test_check_never_shows_first_labels(repo):
    rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    saved, _ = fill(repo)
    _, lines = rl.check()
    out = "\n".join(lines)
    assert all(it["test_ref"] not in out for it in saved["items"])


# ---------- score ----------

def test_score_needs_every_item(repo):
    rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    with pytest.raises(rl.RelabelError, match="blank: B01"):
        rl.score()


def test_score_counts_agreement_kappa_and_runner_ups(repo):
    rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    saved = ls.load_set(rl.NAME)
    first = {it["ref"]: it for it in
             __import__("tomllib").load(open("eval/labels/test.toml", "rb"))["item"]}
    with_vs = [it for it in saved["items"] if rl.runner_up(first[it["test_ref"]]["note"])]
    other = [it for it in saved["items"] if it not in with_vs]
    changes = {with_vs[0]["ref"]: rl.runner_up(first[with_vs[0]["test_ref"]]["note"])}
    for it in other[:2]:
        changes[it["ref"]] = "unclear" if first[it["test_ref"]]["code"] != "unclear" else "other"
    fill(repo, codes=changes)
    result, lines = rl.score()
    assert (result["n"], result["agree"]) == (30, 27)
    assert result["misses_matching_first_runner_up"] == 1
    assert 0 < result["kappa"] < 1 and result["wilson95"][0] < 0.9 < result["wilson95"][1]
    assert lines[0].startswith("Agreement: 27 of 30 (90.0%")
    assert result["gap_days_at_draw"] == 1.0
    assert "Drawn 1.0 days after the first labels (e7efa08)" in "\n".join(lines)
    assert os.path.exists("eval/runs/relabel-score.json")
    assert sum(v["n"] for v in result["by_product"].values()) == 30


def test_uncertain_re_labels_are_still_scored(repo):
    rl.prepare(db_path=str(repo / "complaints.sqlite"), now=NOW)
    fill(repo, edit=lambda t: t.replace("uncertain = false", "uncertain = true", 2))
    result, _ = rl.score()
    assert result["n"] == 30 and result["second_uncertain"] == 2 and result["agree"] == 30


@pytest.mark.parametrize("note, code", [
    ("vs third_party. A deferred plan", "third_party"), ("Vs. fraud_claim, maybe", "fraud_claim"),
    ("Some note without one", None), ("", None), (None, None)])
def test_runner_up_is_read_from_the_note(note, code):
    assert rl.runner_up(note) == code


def test_kappa_matches_a_hand_count():
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]
    assert rl.kappa(pairs) == pytest.approx(0.5)
    assert rl.kappa([("a", "a")] * 3) is None
