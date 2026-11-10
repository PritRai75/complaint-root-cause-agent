"""Taxonomy checks and labelling sets, on a small fake database, plus real-data checks."""
import json
import os
import sqlite3

import pytest

from eval import labelset as ls
from eval import taxonomy as tx
from ingest import build_db, sources
from ingest.textnorm import fingerprint

CARD, CHK = sources.PRODUCTS
MONTHS = ("2026-01", "2026-02", "2026-03")
TEMPLATE = "Please remove the {$35.00} fee charged on XX/XX/XXXX by XXXX."
CAUSES = ("aa_cause", "bb_cause", "cc_cause", "dd_cause", "ee_cause", "ff_cause")


def letters(n):
    return "".join("abcdefghij"[int(d)] for d in str(n))


def make_db(path, per_month=30):
    """per_month narratives for each product and month. Every fifth is the same
    template, so its copies share one fingerprint."""
    conn = sqlite3.connect(path)
    conn.executescript(build_db.SCHEMA)
    cid = 5000
    for product in sources.PRODUCTS:
        for month in MONTHS:
            for i in range(per_month):
                cid += 1
                conn.execute(
                    f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(build_db.DB_COLUMNS))})",
                    (cid, f"{month}-1{i % 9}", product, "Sub", "ISSUE-HIDDEN", None,
                     "BANK HIDDEN", "TX", None, "Web", f"{month}-1{i % 9}",
                     "Closed with monetary relief", None, "Yes", 1, "A.zip"))
                text = (TEMPLATE.replace("35.00", str(i)) if i % 5 == 0
                        else f"Narrative {letters(cid)} about <b>{product}</b> & more.")
                conn.execute("INSERT INTO narratives VALUES (?, ?, ?)", (cid, text, len(text)))
    conn.executemany("INSERT INTO build_info VALUES (?, ?)", [
        ("built_at", "2026-11-02T00:00:00Z"), ("loader_commit", "abc1234"),
        ("complaints_rows", str(cid - 5000)), ("narratives_rows", str(cid - 5000))])
    conn.commit()
    conn.close()


def taxonomy_text(version="draft-1", refs=("R1", "R2")):
    codes = list(CAUSES) + ["third_party", "other", "unclear"]
    out = [f'version = "{version}"', 'written = "2026-11-10"',
           'primary_rule = "Pick the failure the main request would undo."',
           f"precedence = {json.dumps(codes)}", ""]
    for code in codes:
        root = code not in tx.NOT_ROOT_CAUSES
        rule = '{ "Credit card" = "1026.13", "Checking or savings account" = "1005.11" }'
        out += ["[[cause]]", f'code = "{code}"', f'name = "Name {code}"',
                f'definition = "Definition of {code}."', 'signals = "words"',
                'not_this = "near misses"', f"rule = {rule if root else '{}'}",
                f"examples = {json.dumps(list(refs[:1])) if code not in ('other', 'unclear') else '[]'}",
                ""]
    return "\n".join(out)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake database, taxonomy and reading set, with every path pointed at tmp_path."""
    db = str(tmp_path / "fake.sqlite")
    make_db(db)
    conn = sqlite3.connect(db)
    first = [r[0] for r in conn.execute(
        "SELECT complaint_id FROM complaints ORDER BY complaint_id")]
    conn.close()
    # Reading set: the template's lowest ID, so every template copy is excluded,
    # plus one ordinary narrative per product.
    reading = {"items": [
        {"ref": "R1", "complaint_id": first[0], "product": CARD, "month": "2026-01"},
        {"ref": "R2", "complaint_id": first[1], "product": CARD, "month": "2026-01"},
        {"ref": "R3", "complaint_id": first[91], "product": CHK, "month": "2026-01"}]}
    reading_path = tmp_path / "taxonomy_sample.json"
    reading_path.write_text(json.dumps(reading), encoding="utf-8")
    tax_path = tmp_path / "taxonomy.toml"
    tax_path.write_text(taxonomy_text(), encoding="utf-8")

    monkeypatch.setattr(tx, "READING_PATH", str(reading_path))
    monkeypatch.setattr(tx, "TAXONOMY_PATH", str(tax_path))
    monkeypatch.setattr(ls, "SETS_DIR", str(tmp_path / "sets"))
    monkeypatch.setattr(ls, "LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setattr(ls, "SHEETS_DIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(ls, "DRAWS", {"pilot": ("P", 6, 1), "test": ("T", 12, 2)})
    return {"db": db, "tax": tax_path, "reading": reading}


def all_texts(db):
    conn = sqlite3.connect(db)
    texts = dict(conn.execute("SELECT complaint_id, narrative FROM narratives"))
    conn.close()
    return texts


# ---------- taxonomy ----------

def test_good_taxonomy_loads(env):
    tax = tx.load()
    assert tax["codes"][:2] == ["aa_cause", "bb_cause"]
    assert tax["by_code"]["other"]["rule"] == {}


@pytest.mark.parametrize("edit, message", [
    (lambda t: t.replace('"Credit card" = "1026.13"', '"Card" = "1026.13"', 1), "not a product"),
    (lambda t: t.replace('"1005.11"', '"Reg E"', 1), "not a provision"),
    (lambda t: t.replace('code = "aa_cause"', 'code = "AA cause"'), "snake_case"),
    (lambda t: t.replace('"aa_cause", ', "", 1), "precedence must list every code"),
    (lambda t: t.replace('examples = ["R1"]', 'examples = ["R99"]', 1), "not in the reading sample"),
    (lambda t: t.replace('examples = ["R1"]', "examples = []", 1), "at least one example"),
    (lambda t: t.replace('code = "bb_cause"', 'code = "aa_cause"'), "more than once"),
    (lambda t: t.replace('definition = "Definition of cc_cause."', 'definition = " "'),
     "definition is empty"),
    (lambda t: t.replace('version = "draft-1"', 'version = ""'), "version is empty"),
])
def test_taxonomy_problems_are_caught(env, edit, message):
    env["tax"].write_text(edit(taxonomy_text()), encoding="utf-8")
    with pytest.raises(tx.TaxonomyError, match=message):
        tx.load()


def test_taxonomy_counts_root_causes_and_special_codes(env):
    text = taxonomy_text()
    five = text.replace('"ff_cause", ', "")
    five = five[:five.index('[[cause]]\ncode = "ff_cause"')] + \
        five[five.index('[[cause]]\ncode = "third_party"'):]
    env["tax"].write_text(five, encoding="utf-8")
    with pytest.raises(tx.TaxonomyError, match="5 root causes"):
        tx.load()
    no_other = text.replace('code = "other"', 'code = "otherwise"')
    env["tax"].write_text(no_other, encoding="utf-8")
    with pytest.raises(tx.TaxonomyError, match="code other is missing"):
        tx.load()
    ruled = text.replace('code = "unclear"\nname = "Name unclear"\n'
                         'definition = "Definition of unclear."\nsignals = "words"\n'
                         'not_this = "near misses"\nrule = {}',
                         'code = "unclear"\nname = "Name unclear"\n'
                         'definition = "Definition of unclear."\nsignals = "words"\n'
                         'not_this = "near misses"\nrule = { "Credit card" = "1026.13" }')
    env["tax"].write_text(ruled, encoding="utf-8")
    with pytest.raises(tx.TaxonomyError, match="rule must be {} for unclear"):
        tx.load()


# ---------- sets, sheet, label file ----------

def test_pilot_draw_excludes_reading_and_is_seeded(env):
    sample = ls.prepare("pilot", db_path=env["db"])
    texts = all_texts(env["db"])
    ids = [it["complaint_id"] for it in sample["items"]]
    assert len(ids) == len(set(ids)) == 12
    assert sum(it["product"] == CARD for it in sample["items"]) == 6
    reading_fps = {fingerprint(texts[it["complaint_id"]]) for it in env["reading"]["items"]}
    drawn_fps = [fingerprint(texts[i]) for i in ids]
    assert not reading_fps & set(drawn_fps)
    assert fingerprint(TEMPLATE) not in drawn_fps  # every template copy left out
    assert len(set(drawn_fps)) == len(drawn_fps)
    assert [it["ref"] for it in sample["items"]][:2] == ["P01", "P02"]
    assert sample["split"] == "dev" and sample["excluded_sets"] == ["reading"]

    conn = ls.ts.connect_ro(env["db"])
    again = ls.create_set(conn, "pilot", ["reading"], "draft-1")
    conn.close()
    assert again["items"] == sample["items"]


def test_sheet_shows_only_ref_product_and_text(env):
    ls.prepare("pilot", db_path=env["db"])
    page = open(ls.sheet_path("pilot"), encoding="utf-8").read()
    for hidden in ("ISSUE-HIDDEN", "BANK HIDDEN", "monetary relief", "2026-01"):
        assert hidden not in page
    assert "&lt;b&gt;" in page and "<b>Credit card</b>" not in page
    assert page.count("<article") == 12
    assert "Definition of aa_cause." in page


def test_prepare_reuses_the_set_and_never_overwrites_labels(env):
    first = ls.prepare("pilot", db_path=env["db"])
    path = ls.label_path("pilot")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace('code = ""', 'code = "aa_cause"', 1))
    again = ls.prepare("pilot", db_path=env["db"])
    assert again["items"] == first["items"]
    assert 'code = "aa_cause"' in open(path, encoding="utf-8").read()


def test_test_set_waits_for_frozen_taxonomy_and_pilot(env):
    with pytest.raises(ls.LabelError, match="only after the taxonomy is frozen"):
        ls.prepare("test", db_path=env["db"])
    env["tax"].write_text(taxonomy_text(version="v1"), encoding="utf-8")
    with pytest.raises(ls.LabelError, match="draw pilot before test"):
        ls.prepare("test", db_path=env["db"])
    pilot = ls.prepare("pilot", db_path=env["db"])
    test = ls.prepare("test", db_path=env["db"])
    texts = all_texts(env["db"])
    dev_fps = {fingerprint(texts[it["complaint_id"]])
               for it in env["reading"]["items"] + pilot["items"]}
    assert not dev_fps & {fingerprint(texts[it["complaint_id"]]) for it in test["items"]}
    assert test["split"] == "test" and test["excluded_sets"] == ["reading", "pilot"]
    assert test["items"][0]["ref"] == "T01"


def fill(name, edits):
    path = ls.label_path(name)
    with open(path, encoding="utf-8") as f:
        blocks = f.read().split("[[item]]")
    for i, (code, uncertain) in edits.items():
        blocks[i] = blocks[i].replace('code = ""', f'code = "{code}"')
        if uncertain:
            blocks[i] = blocks[i].replace("uncertain = false", "uncertain = true")
    with open(path, "w", encoding="utf-8") as f:
        f.write("[[item]]".join(blocks))


def test_check_counts_progress_and_finds_problems(env):
    ls.prepare("pilot", db_path=env["db"])
    fill("pilot", {1: ("aa_cause", False), 2: ("other", True), 3: ("no_such_code", False)})
    errors, lines = ls.check("pilot")
    assert errors == ["P03: unknown code 'no_such_code'"]
    assert lines[0].startswith("pilot: 2 of 12 labelled under taxonomy draft-1, 9 blank")
    total = [ln for ln in lines if ln.startswith("total")][0].split()
    assert total[-2:] == ["2", "1"]

    env["tax"].write_text(taxonomy_text(version="draft-2"), encoding="utf-8")
    errors, _ = ls.check("pilot")
    assert any("labels made under taxonomy 'draft-1'" in e for e in errors)


def test_check_catches_missing_and_changed_items(env):
    ls.prepare("pilot", db_path=env["db"])
    path = ls.label_path("pilot")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = text.split("[[item]]")
    blocks[2] = blocks[2].replace(f'product = "{CARD}"', f'product = "{CHK}"') \
                         .replace(f'product = "{CHK}"', 'product = "Mortgage"')
    text = "[[item]]".join(blocks[:-1])  # drop the last item
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    errors, _ = ls.check("pilot")
    assert "P12 is missing" in errors
    assert "P02: complaint_id or product differs from the saved set" in errors


def test_reading_set_gets_a_label_file(env):
    ls.prepare("reading", db_path=env["db"])
    errors, lines = ls.check("reading")
    assert errors == []
    assert lines[0].startswith("reading: 0 of 3 labelled")


# ---------- real data (skipped in CI and without the database) ----------

def test_committed_taxonomy_is_well_formed():
    if not os.path.exists("eval/taxonomy.toml"):
        pytest.skip("taxonomy not written yet")
    tax = tx.load("eval/taxonomy.toml", "eval/taxonomy_sample.json")
    assert {"third_party", "other", "unclear"} <= set(tax["codes"])


@pytest.mark.skipif(not os.path.exists(sources.DB_PATH), reason="database not on this machine")
@pytest.mark.parametrize("name", ["pilot", "test"])
def test_committed_sets_reproduce(name):
    path = os.path.join("eval/sets", f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"{name} set not drawn yet")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    conn = ls.ts.connect_ro(sources.DB_PATH)
    fresh = ls.create_set(conn, name, saved["excluded_sets"], saved["taxonomy_version_at_draw"])
    conn.close()
    assert fresh["items"] == saved["items"]
    assert fresh["allocation"] == saved["allocation"]
