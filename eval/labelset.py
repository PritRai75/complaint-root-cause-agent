"""Labelling sets: draw them, write the labelling sheet and label file, check labels.

Usage:
  python -m eval.labelset prepare NAME   draw the set if it has no saved IDs yet,
                                         then write the sheet, and the label file
                                         if it doesn't exist yet
  python -m eval.labelset check NAME     check the labels and print the spread

NAME is one of:
  reading  the 60 narratives read to draft the taxonomy (eval/taxonomy_sample.json,
           already drawn). Dev.
  pilot    15 fresh narratives per product, labelled cold from the written
           definitions to test them before the taxonomy is frozen. Dev.
  test     100 per product, labelled under the frozen taxonomy. Its numbers are
           the published ones. Drawn only once the taxonomy version starts with "v".

Every draw keeps one narrative per distinct text (see ingest/textnorm.py) and
leaves out every text in a set drawn before it, so the same text never sits
in two sets and nothing read while designing the taxonomy reaches test.
Within a product, months get a share in proportion to their distinct texts,
as in eval/taxonomy_sample.py. The drawn IDs are saved to eval/sets/NAME.json
and committed. Once saved, a set is never drawn again.

The labelling sheet (scratch/labels_NAME.html, not committed) shows the ref,
the product and the narrative, nothing else: no issue, sub-issue, company,
company response or month. The tagger gets the same inputs. The label file
(eval/labels/NAME.toml) is Prit's and is never overwritten by this script.
"""
import html
import json
import os
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone

from eval import taxonomy as tx
from eval import taxonomy_sample as ts
from ingest import sources
from ingest.build_db import git_commit
from ingest.textnorm import fingerprint

SETS_DIR = "eval/sets"
LABELS_DIR = "eval/labels"
SHEETS_DIR = "scratch"

# name -> (ref prefix, per product, seed). reading is not drawn here.
DRAWS = {
    "pilot": ("P", 15, 20261110),
    "test": ("T", 100, 20261111),
}
ORDER = ("reading", "pilot", "test")


class LabelError(Exception):
    pass


def set_path(name):
    if name == "reading":
        return tx.READING_PATH
    return os.path.join(SETS_DIR, f"{name}.json")


def label_path(name):
    return os.path.join(LABELS_DIR, f"{name}.toml")


def sheet_path(name):
    return os.path.join(SHEETS_DIR, f"labels_{name}.html")


def load_set(name):
    path = set_path(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def saved_sets():
    return [n for n in ORDER if load_set(n) is not None]


def texts_for(conn, ids):
    ids = list(ids)
    found = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        marks = ", ".join("?" * len(chunk))
        found.update(conn.execute(
            f"SELECT complaint_id, narrative FROM narratives WHERE complaint_id IN ({marks})",
            chunk))
    missing = [c for c in ids if c not in found]
    if missing:
        raise LabelError(f"IDs not in this database: {missing[:5]}. "
                         "Was it rebuilt from different files?")
    return found


def create_set(conn, name, excluded_names, tax_version):
    prefix, per_product, seed = DRAWS[name]
    excluded_ids = [it["complaint_id"] for n in excluded_names for it in load_set(n)["items"]]
    excluded_fps = {fingerprint(t) for t in texts_for(conn, excluded_ids).values()}
    pool, _ = ts.scan(conn)
    texts = texts_for(conn, [cid for ids in pool.values() for cid in ids])
    pool = {k: [cid for cid in ids if fingerprint(texts[cid]) not in excluded_fps]
            for k, ids in pool.items()}
    pool = {k: ids for k, ids in pool.items() if ids}
    allocation, items = ts.draw(pool, sources.PRODUCTS, per_product, seed)
    width = len(str(len(items)))
    for i, it in enumerate(items, 1):
        it["ref"] = f"{prefix}{i:0{width}d}"
    info = ts.build_info(conn)
    return {
        "set": name,
        "split": "test" if name == "test" else "dev",
        "pool": "one narrative per distinct fingerprint (lowest complaint ID), "
                "leaving out every fingerprint in the excluded sets",
        "excluded_sets": list(excluded_names),
        "excluded_fingerprints": len(excluded_fps),
        "seed": seed,
        "per_product": per_product,
        "taxonomy_version_at_draw": tax_version,
        "drawn_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": git_commit(),
        "database": {k: info.get(k) for k in
                     ("built_at", "loader_commit", "complaints_rows", "narratives_rows")},
        "allocation": allocation,
        "items": items,
    }


def skeleton(name, sample, tax_version):
    q = json.dumps  # a JSON string is a valid TOML basic string
    out = [
        f"# Labels for the {name} set ({len(sample['items'])} narratives). Written by Prit.",
        "#",
        "# For each item set code to one code from eval/taxonomy.toml, chosen by its",
        "# primary_rule. uncertain = true leaves the item out of scoring. note is",
        "# optional: one line on why, or what made it hard.",
        "# Check with: python -m eval.labelset check " + name,
        "",
        f"set = {q(name)}",
        f"taxonomy_version = {q(tax_version)}   # the version these labels were made under",
        "",
    ]
    for it in sample["items"]:
        out += ["[[item]]",
                f"ref = {q(it['ref'])}",
                f"complaint_id = {it['complaint_id']}",
                f"product = {q(it['product'])}",
                'code = ""',
                "uncertain = false",
                'note = ""',
                ""]
    return "\n".join(out)


SHEET_CSS = ts.SHEET_CSS + """
.codes td { vertical-align:top; font-size:.85rem; }
.codes td:first-child { font-family:ui-monospace, monospace; white-space:nowrap; }
"""


def render_sheet(name, sample, texts, tax):
    e = html.escape
    items = sample["items"]
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Labelling sheet {e(name)}</title><style>{SHEET_CSS}</style></head><body><main>",
        f"<h1>Labelling sheet: {e(name)}</h1>",
        f"<p class='lede'>{len(items)} narratives. Each shows only the product and the text, "
        "the same inputs the tagger gets. Label in "
        f"<code>{e(label_path(name))}</code> under taxonomy {e(tax['version'])}. "
        "Narratives often name the bank even though the company field is hidden.</p>",
        "<table class='codes'><tr><th>Code</th><th>Definition</th></tr>",
    ]
    for code in tax["precedence"]:
        c = tax["by_code"][code]
        out.append(f"<tr><td>{e(code)}</td><td><b>{e(c['name'])}</b>. "
                   f"{e(' '.join(c['definition'].split()))}</td></tr>")
    out.append("</table><nav>")
    out += [f"<a href='#{it['ref']}'>{it['ref']}</a>" for it in items]
    out.append("</nav>")
    for it in items:
        text = texts[it["complaint_id"]]
        out += [f"<article id='{it['ref']}'><h2>{it['ref']} · {e(it['product'])}</h2>",
                f"<p class='meta'>{len(text):,} characters</p>",
                f"<div class='text'>{e(text)}</div></article>"]
    out.append("</main></body></html>")
    return "\n".join(out)


def write_new(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")


def prepare(name, db_path=sources.DB_PATH, taxonomy_path=None):
    if name not in ORDER:
        raise LabelError(f"unknown set {name!r}; expected one of {', '.join(ORDER)}")
    tax = tx.load(taxonomy_path)
    conn = ts.connect_ro(db_path)
    try:
        sample = load_set(name)
        if sample is not None:
            print(f"Using the saved set in {set_path(name)} (no new draw)")
        elif name == "reading":
            raise LabelError(f"{set_path(name)} not found. Run python -m eval.taxonomy_sample")
        else:
            if name == "test" and not tax["version"].startswith("v"):
                raise LabelError(f"taxonomy version is {tax['version']!r}. Draw the test set "
                                 "only after the taxonomy is frozen (version v1 or later)")
            earlier = ORDER[:ORDER.index(name)]
            missing = [n for n in earlier if load_set(n) is None]
            if missing:
                raise LabelError(f"draw {', '.join(missing)} before {name}")
            sample = create_set(conn, name, earlier, tax["version"])
            write_new(set_path(name), json.dumps(sample, indent=1))
            print(f"Drew {len(sample['items'])} narratives for {name}, "
                  f"saved their IDs to {set_path(name)}")
            for a in sample["allocation"]:
                print(f"  {a['product']:<28} {a['month']}  distinct {a['pool']:>6,}  "
                      f"drawn {a['drawn']:>3}")
        texts = texts_for(conn, [it["complaint_id"] for it in sample["items"]])
    finally:
        conn.close()

    write_new(sheet_path(name), render_sheet(name, sample, texts, tax))
    print(f"Wrote the labelling sheet to {sheet_path(name)}")
    if os.path.exists(label_path(name)):
        print(f"{label_path(name)} already exists, left as it is")
    else:
        write_new(label_path(name), skeleton(name, sample, tax["version"]))
        print(f"Wrote an empty label file to {label_path(name)}")
    return sample


def check(name, taxonomy_path=None):
    """Return (errors, summary lines). Blank codes are progress, not errors."""
    tax = tx.load(taxonomy_path)
    sample = load_set(name)
    if sample is None:
        raise LabelError(f"no saved set {name!r}. Run python -m eval.labelset prepare {name}")
    path = label_path(name)
    if not os.path.exists(path):
        raise LabelError(f"{path} not found. Run python -m eval.labelset prepare {name}")
    with open(path, "rb") as f:
        try:
            labels = tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            raise LabelError(f"{path} is not valid TOML: {err}") from err

    errors = []
    if labels.get("set") != name:
        errors.append(f"set = {labels.get('set')!r}, expected {name!r}")
    if labels.get("taxonomy_version") != tax["version"]:
        errors.append(f"labels made under taxonomy {labels.get('taxonomy_version')!r}, "
                      f"but eval/taxonomy.toml is {tax['version']!r}. Review every label "
                      "under the new version, then update taxonomy_version")
    expected = {it["ref"]: it for it in sample["items"]}
    items = labels.get("item", [])
    seen = Counter(it.get("ref") for it in items)
    for ref, n in seen.items():
        if n > 1:
            errors.append(f"{ref} appears {n} times")
    for ref in expected.keys() - seen.keys():
        errors.append(f"{ref} is missing")
    for ref in seen.keys() - expected.keys():
        errors.append(f"{ref} is not in the {name} set")
    blank, rows = [], []
    for it in items:
        ref = it.get("ref")
        want = expected.get(ref)
        if want is None:
            continue
        if it.get("complaint_id") != want["complaint_id"] or it.get("product") != want["product"]:
            errors.append(f"{ref}: complaint_id or product differs from the saved set")
        code = it.get("code", "")
        if not isinstance(it.get("uncertain"), bool):
            errors.append(f"{ref}: uncertain must be true or false")
        if code == "":
            blank.append(ref)
        elif code not in tax["codes"]:
            errors.append(f"{ref}: unknown code {code!r}")
        else:
            rows.append((code, it["product"], bool(it.get("uncertain"))))

    card, chk = sources.PRODUCTS
    lines = [f"{name}: {len(rows)} of {len(expected)} labelled under taxonomy "
             f"{labels.get('taxonomy_version')}"
             + (f", {len(blank)} blank ({' '.join(blank[:10])}{' ...' if len(blank) > 10 else ''})"
                if blank else ""),
             "",
             f"{'code':<20} {'card':>5} {'checking':>9} {'total':>6} {'uncertain':>10}"]
    for code in tax["precedence"]:
        n_card = sum(1 for c, p, _ in rows if c == code and p == card)
        n_chk = sum(1 for c, p, _ in rows if c == code and p == chk)
        n_unc = sum(1 for c, _, u in rows if c == code and u)
        lines.append(f"{code:<20} {n_card:>5} {n_chk:>9} {n_card + n_chk:>6} {n_unc:>10}")
    n_unc = sum(1 for *_, u in rows if u)
    lines.append(f"{'total':<20} {sum(p == card for _, p, _ in rows):>5} "
                 f"{sum(p == chk for _, p, _ in rows):>9} {len(rows):>6} {n_unc:>10}")
    return errors, lines


def main(argv):
    if len(argv) != 2 or argv[0] not in ("prepare", "check"):
        sys.exit("usage: python -m eval.labelset prepare|check reading|pilot|test")
    action, name = argv
    try:
        if action == "prepare":
            prepare(name)
        else:
            errors, lines = check(name)
            print("\n".join(lines))
            if errors:
                print(f"\n{len(errors)} problem(s):")
                print("\n".join(f"  {e}" for e in errors))
                sys.exit(1)
    except (LabelError, tx.TaxonomyError, ts.SampleError) as err:
        sys.exit(f"ERROR {err}")


if __name__ == "__main__":
    main(sys.argv[1:])
