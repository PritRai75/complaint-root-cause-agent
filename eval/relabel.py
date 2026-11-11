"""Blind re-label: Prit labels 30 test narratives a second time, without seeing
the first labels. The result is human self-agreement,
the context for the tagger's agreement on test (decision 20).

Usage:
  python -m eval.relabel prepare   draw the 30 (once), write the sheet
                                   scratch/labels_relabel.html and the empty label
                                   file eval/labels/relabel.toml
  python -m eval.relabel check     check the re-labels; shows nothing from the first labels
  python -m eval.relabel score     only when all 30 are coded: compare with the first labels

Rules, fixed before any result (S5 decision 17; S6, Prit):
  - 30 items from the test set, 15 per product, by a seeded random draw (no
    stratification by code).
  - The 14 test items discussed by reference in session 3 are left out, since
    their labels can't be re-labelled blind (Project 1's rule).
  - No waiting period (Prit, 11 Nov, replacing "from 17 Nov"). The gap since the
    first labels (e7efa08, 10 Nov 22:09 IST) is recorded at the draw and printed
    with the score. After a gap of a day or two, memory of the first labels can
    raise agreement, so read the figure as an upper bound and state the gap.
  - The sheet shows only a new ref (B01 to B30, shuffled), the product and the
    text: no T ref, no complaint ID, no first code or note. The label file has
    no complaint ID either. The mapping is in eval/sets/relabel.json, which
    the scorer reads: don't open it, or eval/labels/test.toml, until you've
    finished.
  - Score: exact agreement over all 30 (uncertain items are not left out),
    with a Wilson 95% interval and Cohen's kappa. Disagreements are listed with
    the first label's "vs" runner-up, and how many second codes match it is
    counted as a description, not a score.
"""
import json
import os
import random
import re
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone

from eval import labelset as ls
from eval import taxonomy as tx
from eval.tagrun import wilson
from ingest import sources

NAME = "relabel"
SOURCE = "test"
PER_PRODUCT = 15
SEED = 20261117
FIRST_LABELS_COMMIT = "e7efa08"
FIRST_LABELS_AT = datetime(2026, 11, 10, 16, 39, 43, tzinfo=timezone.utc)   # 22:09 IST
PREFIX = "B"
# Discussed by reference with Prit in session 3 (claude/session-3/handoff.md).
DISCUSSED = ("T022", "T024", "T041", "T061", "T065", "T071", "T116", "T126", "T154",
             "T157", "T160", "T161", "T166", "T193")
RUNS_DIR = "eval/runs"


class RelabelError(Exception):
    pass


def draw(test_set, per_product=PER_PRODUCT, seed=SEED, excluded=DISCUSSED):
    """[{ref, test_ref, complaint_id, product}], shuffled, refs B01 onwards."""
    rng = random.Random(seed)
    chosen = []
    for product in sources.PRODUCTS:
        pool = sorted((it for it in test_set["items"]
                       if it["product"] == product and it["ref"] not in excluded),
                      key=lambda it: it["ref"])
        if len(pool) < per_product:
            raise RelabelError(f"only {len(pool)} {product} items to draw from")
        chosen += rng.sample(pool, per_product)
    rng.shuffle(chosen)
    width = len(str(len(chosen)))
    return [{"ref": f"{PREFIX}{i:0{width}d}", "test_ref": it["ref"],
             "complaint_id": it["complaint_id"], "product": it["product"]}
            for i, it in enumerate(chosen, 1)]


def skeleton(items, tax_version):
    q = json.dumps
    out = [f"# Blind re-label ({len(items)} narratives). Written by Prit.",
           "#",
           "# Label from the sheet scratch/labels_relabel.html, exactly as the first time:",
           "# one code by the primary_rule, uncertain = true only for a genuine coin-flip,",
           "# a note when torn ('vs <code>', then why). Don't open eval/labels/test.toml or",
           "# eval/sets/relabel.json until all items are done.",
           "# Check with: python -m eval.relabel check",
           "",
           f"set = {q(NAME)}",
           f"taxonomy_version = {q(tax_version)}",
           ""]
    for it in items:
        out += ["[[item]]", f"ref = {q(it['ref'])}", f"product = {q(it['product'])}",
                'code = ""', "uncertain = false", 'note = ""', ""]
    return "\n".join(out)


def gap_days(when):
    return round((when - FIRST_LABELS_AT).total_seconds() / 86400, 2)


def prepare(db_path=sources.DB_PATH, now=None, taxonomy_path=None):
    now = now or datetime.now(timezone.utc)
    tax = tx.load(taxonomy_path)
    saved = ls.load_set(NAME)
    if saved is None:
        test_set = ls.load_set(SOURCE)
        items = draw(test_set)
        saved = {"set": NAME, "source": SOURCE, "per_product": PER_PRODUCT, "seed": SEED,
                 "excluded_refs": list(DISCUSSED),
                 "excluded_why": "discussed by reference in session 3",
                 "drawn_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "first_labels_commit": FIRST_LABELS_COMMIT,
                 "gap_days_at_draw": gap_days(now),
                 "code_commit": ls.git_commit(), "items": items}
        ls.write_new(ls.set_path(NAME), json.dumps(saved, indent=1))
        print(f"Drew {len(items)} items, saved to {ls.set_path(NAME)} (don't open it). "
              f"{saved['gap_days_at_draw']} days after the first labels.")
    else:
        print(f"Using the saved draw in {ls.set_path(NAME)} (no new draw)")
    conn = ls.ts.connect_ro(db_path)
    try:
        texts = ls.texts_for(conn, [it["complaint_id"] for it in saved["items"]])
    finally:
        conn.close()
    blind = {"items": [{k: it[k] for k in ("ref", "complaint_id", "product")}
                       for it in saved["items"]]}
    ls.write_new(ls.sheet_path(NAME), ls.render_sheet(NAME, blind, texts, tax))
    print(f"Wrote the sheet to {ls.sheet_path(NAME)}")
    if os.path.exists(ls.label_path(NAME)):
        print(f"{ls.label_path(NAME)} already exists, left as it is")
    else:
        ls.write_new(ls.label_path(NAME), skeleton(saved["items"], tax["version"]))
        print(f"Wrote an empty label file to {ls.label_path(NAME)}")
    return saved


def load_relabels(taxonomy_path=None):
    """(saved draw, labels by B ref, problems, blank refs)."""
    tax = tx.load(taxonomy_path)
    saved = ls.load_set(NAME)
    if saved is None:
        raise RelabelError("no draw yet. Run python -m eval.relabel prepare")
    path = ls.label_path(NAME)
    if not os.path.exists(path):
        raise RelabelError(f"{path} not found. Run python -m eval.relabel prepare")
    with open(path, "rb") as f:
        try:
            labels = tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            raise RelabelError(f"{path} is not valid TOML: {err}") from err
    problems, blank, by_ref = [], [], {}
    if labels.get("set") != NAME:
        problems.append(f"set = {labels.get('set')!r}, expected {NAME!r}")
    if labels.get("taxonomy_version") != tax["version"]:
        problems.append(f"taxonomy_version {labels.get('taxonomy_version')!r}, "
                        f"expected {tax['version']!r}")
    expected = {it["ref"]: it for it in saved["items"]}
    counts = Counter(it.get("ref") for it in labels.get("item", []))
    problems += [f"{r} appears {n} times" for r, n in counts.items() if n > 1]
    problems += [f"{r} is missing" for r in sorted(expected.keys() - counts.keys())]
    problems += [f"{r} is not in the draw" for r in sorted(counts.keys() - expected.keys())]
    for it in labels.get("item", []):
        ref = it.get("ref")
        if ref not in expected:
            continue
        if it.get("product") != expected[ref]["product"]:
            problems.append(f"{ref}: product differs from the draw")
        if not isinstance(it.get("uncertain"), bool):
            problems.append(f"{ref}: uncertain must be true or false")
        code = it.get("code", "")
        if code == "":
            blank.append(ref)
        elif code not in tax["codes"]:
            problems.append(f"{ref}: unknown code {code!r}")
        by_ref[ref] = it
    return saved, by_ref, problems, sorted(blank)


def check(taxonomy_path=None):
    saved, by_ref, problems, blank = load_relabels(taxonomy_path)
    done = len(saved["items"]) - len(blank) - sum(1 for p in problems if "unknown code" in p)
    lines = [f"relabel: {done} of {len(saved['items'])} coded"
             + (f", blank: {' '.join(blank)}" if blank else "")]
    return problems, lines


def runner_up(note):
    m = re.match(r"\s*vs\.?\s+([a-z_]+)", note or "", re.IGNORECASE)
    return m.group(1).lower() if m else None


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(first[c] * second[c] for c in first) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(taxonomy_path=None, labels_dir=ls.LABELS_DIR):
    saved, by_ref, problems, blank = load_relabels(taxonomy_path)
    if problems or blank:
        raise RelabelError("fix the re-label file first: "
                           + "; ".join(problems + ([f"blank: {' '.join(blank)}"] if blank else [])))
    with open(os.path.join(labels_dir, f"{SOURCE}.toml"), "rb") as f:
        first = {it["ref"]: it for it in tomllib.load(f)["item"]}
    rows = []
    for it in saved["items"]:
        a, b = first[it["test_ref"]], by_ref[it["ref"]]
        rows.append({"ref": it["ref"], "test_ref": it["test_ref"], "product": it["product"],
                     "first": a["code"], "second": b["code"],
                     "first_runner_up": runner_up(a.get("note")),
                     "second_uncertain": b["uncertain"]})
    n = len(rows)
    agree = sum(r["first"] == r["second"] for r in rows)
    low, high = wilson(agree, n)
    misses = [r for r in rows if r["first"] != r["second"]]
    k = kappa([(r["first"], r["second"]) for r in rows])
    result = {
        "set": NAME, "n": n, "agree": agree, "share": round(agree / n, 4),
        "wilson95": [round(low, 4), round(high, 4)],
        "kappa": None if k is None else round(k, 4),
        "by_product": {p: {"n": sum(r["product"] == p for r in rows),
                           "agree": sum(r["product"] == p and r["first"] == r["second"]
                                        for r in rows)} for p in sources.PRODUCTS},
        "misses_matching_first_runner_up": sum(r["second"] == r["first_runner_up"]
                                               for r in misses),
        "second_uncertain": sum(r["second_uncertain"] for r in rows),
        "gap_days_at_draw": saved.get("gap_days_at_draw"),
        "rows": rows,
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, "relabel-score.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    lines = [f"Agreement: {agree} of {n} ({100 * agree / n:.1f}%, Wilson 95% "
             f"{100 * low:.1f} to {100 * high:.1f}%), Cohen's kappa "
             f"{result['kappa'] if result['kappa'] is not None else 'n/a'}"]
    for p, v in result["by_product"].items():
        lines.append(f"  {p:<28} {v['agree']} of {v['n']}")
    lines.append(f"Drawn {result['gap_days_at_draw']} days after the first labels "
                 f"({FIRST_LABELS_COMMIT}). A short gap can raise agreement: read it as an "
                 "upper bound.")
    if misses:
        lines.append(f"\n{len(misses)} disagreements ({result['misses_matching_first_runner_up']} "
                     "match the first label's 'vs' runner-up; a description, not a score):")
        for r in misses:
            lines.append(f"  {r['ref']} = {r['test_ref']}  first {r['first']:<20} second "
                         f"{r['second']:<20} first runner-up {r['first_runner_up'] or '-'}")
    if result["second_uncertain"]:
        lines.append(f"\n{result['second_uncertain']} re-label(s) marked uncertain "
                     "(still scored, as decided).")
    lines.append(f"\nSaved to {path}")
    return result, lines


def main(argv):
    if len(argv) != 1 or argv[0] not in ("prepare", "check", "score"):
        sys.exit("usage: python -m eval.relabel prepare|check|score")
    try:
        if argv[0] == "prepare":
            prepare()
        elif argv[0] == "check":
            problems, lines = check()
            print("\n".join(lines))
            if problems:
                print(f"\n{len(problems)} problem(s):\n" + "\n".join(f"  {p}" for p in problems))
                sys.exit(1)
        else:
            _, lines = score()
            print("\n".join(lines))
    except (RelabelError, ls.LabelError, tx.TaxonomyError) as err:
        sys.exit(f"ERROR {err}")


if __name__ == "__main__":
    main(sys.argv[1:])
