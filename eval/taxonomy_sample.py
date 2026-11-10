"""Draw the narratives Prit reads to draft the root-cause taxonomy.

Usage:
  python -m eval.taxonomy_sample           draw once, then write the reading sheet
  python -m eval.taxonomy_sample --force   draw again and replace the saved sample

The draw is a seeded random sample of PER_PRODUCT narratives from each
product. Template complaints copied almost word for word would waste reading
slots, so the pool keeps one narrative per distinct text (same fingerprint,
see ingest/textnorm.py), the one with the lowest complaint ID. The sheet says
how many other narratives share each text. Within a product, each month gets
a share in proportion to how many distinct narratives it has
(largest-remainder rounding), so August, which has none, gets none.

The complaint IDs are saved to eval/taxonomy_sample.json and committed. Every
narrative in it, and every narrative with the same fingerprint, goes to the
dev split later, never test, because reading it shaped the taxonomy.

Once the IDs file exists the script does not draw again. It only rebuilds the
reading sheet (scratch/taxonomy_sample.html, not committed) from the saved IDs.

The sheet leaves out the company field (narratives often name the bank anyway).
The labelling sheets in eval/labelset.py leave it out too.
"""
import html
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone

from ingest import sources
from ingest.build_db import git_commit
from ingest.textnorm import fingerprint

SEED = 20261108
PER_PRODUCT = 30
SAMPLE_PATH = "eval/taxonomy_sample.json"
SHEET_PATH = "scratch/taxonomy_sample.html"


class SampleError(Exception):
    pass


def connect_ro(db_path):
    if not os.path.exists(db_path):
        raise SampleError(f"{db_path} not found. Run python -m ingest.build_db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def scan(conn):
    """Return (pool, copies).

    pool maps (product, month) to the sorted complaint IDs that stand for a
    distinct text, the lowest ID per fingerprint across the whole scope.
    copies counts how many in-scope narratives share each fingerprint.
    """
    copies, first = {}, {}
    rows = conn.execute(
        "SELECT c.complaint_id, c.product, substr(c.date_received, 1, 7), n.narrative "
        "FROM complaints c JOIN narratives n USING (complaint_id) "
        "ORDER BY c.complaint_id")
    for cid, product, month, text in rows:
        fp = fingerprint(text)
        copies[fp] = copies.get(fp, 0) + 1
        if fp not in first:
            first[fp] = (cid, product, month)
    pool = {}
    for cid, product, month in sorted(first.values()):
        pool.setdefault((product, month), []).append(cid)
    return pool, Counter(copies)


def allocate(counts, n):
    """Split n across keys in proportion to counts, by largest remainder.

    Ties on the remainder go to the key that sorts first, so the result
    depends only on the counts.
    """
    total = sum(counts.values())
    if n > total:
        raise SampleError(f"asked for {n} from a pool of {total}")
    quota = {k: n * c / total for k, c in counts.items()}
    share = {k: int(q) for k, q in quota.items()}
    left = n - sum(share.values())
    for k in sorted(counts, key=lambda k: (-(quota[k] - share[k]), k))[:left]:
        share[k] += 1
    return share


def draw(pool, products, per_product, seed):
    """Return (allocation rows, items in reading order)."""
    rng = random.Random(seed)
    allocation, picked = [], []
    for product in products:
        counts = {m: len(ids) for (p, m), ids in pool.items() if p == product}
        if not counts:
            raise SampleError(f"no narratives for {product!r}")
        share = allocate(counts, per_product)
        for month in sorted(counts):
            k = share[month]
            allocation.append({"product": product, "month": month,
                               "pool": counts[month], "drawn": k})
            for cid in rng.sample(pool[(product, month)], k):
                picked.append({"complaint_id": cid, "product": product, "month": month})
    rng.shuffle(picked)  # mix products and months so reading isn't in blocks
    width = len(str(len(picked)))
    for i, item in enumerate(picked, 1):
        item["ref"] = f"R{i:0{width}d}"
    return allocation, [{"ref": it["ref"], "complaint_id": it["complaint_id"],
                         "product": it["product"], "month": it["month"]} for it in picked]


def build_info(conn):
    return dict(conn.execute("SELECT key, value FROM build_info"))


def create_sample(conn, products=sources.PRODUCTS, per_product=PER_PRODUCT, seed=SEED):
    pool, _ = scan(conn)
    allocation, items = draw(pool, products, per_product, seed)
    info = build_info(conn)
    return {
        "purpose": "Read to draft the root-cause taxonomy. Every complaint here, and every "
                   "narrative with the same fingerprint, goes to the dev split, never test.",
        "pool": "one narrative per distinct fingerprint (lowest complaint ID)",
        "seed": seed,
        "per_product": per_product,
        "drawn_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": git_commit(),
        "database": {k: info.get(k) for k in
                     ("built_at", "loader_commit", "complaints_rows", "narratives_rows")},
        "allocation": allocation,
        "items": items,
    }


def load_details(conn, items):
    ids = [it["complaint_id"] for it in items]
    marks = ", ".join("?" * len(ids))
    rows = conn.execute(
        "SELECT c.complaint_id, c.sub_product, c.issue, c.sub_issue, c.company_response, "
        "c.timely_response, c.submitted_via, n.narrative, n.char_length "
        f"FROM complaints c JOIN narratives n USING (complaint_id) "
        f"WHERE c.complaint_id IN ({marks})", ids).fetchall()
    found = {r[0]: r for r in rows}
    missing = [cid for cid in ids if cid not in found]
    if missing:
        raise SampleError(f"sample IDs not in this database: {missing[:5]}. "
                          "Was it rebuilt from different files?")
    keys = ("complaint_id", "sub_product", "issue", "sub_issue", "company_response",
            "timely_response", "submitted_via", "narrative", "char_length")
    return {cid: dict(zip(keys, row)) for cid, row in found.items()}


SHEET_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a64; --card:#ffffff; --line:#e3e1da;
        --accent:#7a4b12; --flag:#fff3d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161614; --fg:#ecebe6; --muted:#a3a29b; --card:#1f1f1c; --line:#34332f;
          --accent:#e0b46a; --flag:#3a2f17; } }
body { background:var(--bg); color:var(--fg); margin:0;
       font:16px/1.55 Georgia, "Times New Roman", serif; }
main { max-width:52rem; margin:0 auto; padding:2rem 1rem 4rem; }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
.lede, .meta, td, th, .tag { font-family:system-ui, sans-serif; }
.lede { color:var(--muted); font-size:.95rem; }
table { border-collapse:collapse; font-size:.85rem; margin:.75rem 0 1.5rem; }
td, th { border-bottom:1px solid var(--line); padding:.2rem .7rem .2rem 0; text-align:left; }
td.n { text-align:right; font-variant-numeric:tabular-nums; }
nav { font-family:system-ui, sans-serif; font-size:.85rem; line-height:2; }
nav a { color:var(--accent); margin-right:.6rem; text-decoration:none; }
article { background:var(--card); border:1px solid var(--line); border-radius:6px;
          padding:1rem 1.2rem; margin:1.25rem 0; }
article h2 { font:600 1.05rem system-ui, sans-serif; margin:0 0 .35rem; }
.meta { color:var(--muted); font-size:.85rem; margin:0 0 .2rem; }
.meta b { color:var(--fg); font-weight:600; }
.tag { display:inline-block; background:var(--flag); border-radius:4px; padding:0 .4rem;
       font-size:.8rem; margin:.3rem .3rem 0 0; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; margin-top:.8rem; }
"""


def _month_label(month):
    return datetime.strptime(month, "%Y-%m").strftime("%b %Y")


def render_sheet(sample, details, copies):
    e = html.escape
    items = sample["items"]

    db = sample["database"]
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Taxonomy reading sample</title><style>{SHEET_CSS}</style></head><body><main>",
        "<h1>Taxonomy reading sample</h1>",
        f"<p class='lede'>{len(items)} narratives, seed {sample['seed']}, "
        f"{sample['per_product']} per product, one per distinct text, months in proportion "
        "to their counts of distinct narratives. "
        f"Drawn {e(sample['drawn_at'])} from the database built {e(str(db['built_at']))} "
        f"(loader {e(str(db['loader_commit']))}). The company field is left out, "
        "though narratives often name the bank. "
        "Every complaint here goes to the dev split.</p>",
        "<p class='lede'>Complaints may cite the 2024 $8 late-fee rule (vacated April 2025) "
        "or the 2024 overdraft rule (overturned May 2025). Neither is law.</p>",
        "<table><tr><th>Product</th><th>Month</th><th>Distinct narratives</th><th>Drawn</th></tr>",
    ]
    for a in sample["allocation"]:
        out.append(f"<tr><td>{e(a['product'])}</td><td>{_month_label(a['month'])}</td>"
                   f"<td class='n'>{a['pool']:,}</td><td class='n'>{a['drawn']}</td></tr>")
    out.append("</table><nav>")
    out += [f"<a href='#{it['ref']}'>{it['ref']}</a>" for it in items]
    out.append("</nav>")

    for it in items:
        d = details[it["complaint_id"]]
        fp = fingerprint(d["narrative"])
        tags = []
        if copies[fp] > 1:
            tags.append(f"same text as {copies[fp] - 1:,} other narrative"
                        f"{'s' if copies[fp] > 2 else ''} in scope")
        out += [
            f"<article id='{it['ref']}'><h2>{it['ref']} · {e(it['product'])} · "
            f"{_month_label(it['month'])}</h2>",
            f"<p class='meta'>Complaint {it['complaint_id']} · {d['char_length']:,} characters · "
            f"via {e(d['submitted_via'] or 'n/a')}</p>",
            f"<p class='meta'><b>Sub-product</b> {e(d['sub_product'] or 'none')}</p>",
            f"<p class='meta'><b>Issue</b> {e(d['issue'])} · <b>Sub-issue</b> "
            f"{e(d['sub_issue'] or 'none')}</p>",
            f"<p class='meta'><b>Company response</b> {e(d['company_response'] or 'none')} · "
            f"<b>Timely</b> {e(d['timely_response'] or 'blank')}</p>",
        ]
        out += [f"<span class='tag'>{e(t)}</span>" for t in tags]
        out.append(f"<div class='text'>{e(d['narrative'])}</div></article>")
    out.append("</main></body></html>")
    return "\n".join(out)


def run(db_path=sources.DB_PATH, sample_path=SAMPLE_PATH, sheet_path=SHEET_PATH, force=False):
    conn = connect_ro(db_path)
    try:
        if os.path.exists(sample_path) and not force:
            with open(sample_path, encoding="utf-8") as f:
                sample = json.load(f)
            print(f"Using the saved sample in {sample_path} (no new draw)")
        else:
            sample = create_sample(conn)
            os.makedirs(os.path.dirname(sample_path) or ".", exist_ok=True)
            with open(sample_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(sample, f, indent=1)
                f.write("\n")
            print(f"Drew {len(sample['items'])} narratives, saved their IDs to {sample_path}")
        details = load_details(conn, sample["items"])
        _, copies = scan(conn)
    finally:
        conn.close()

    os.makedirs(os.path.dirname(sheet_path) or ".", exist_ok=True)
    with open(sheet_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_sheet(sample, details, copies))
    print(f"Wrote the reading sheet to {sheet_path}")
    for a in sample["allocation"]:
        print(f"  {a['product']:<28} {a['month']}  distinct {a['pool']:>6,}  drawn {a['drawn']:>2}")
    n_all, n_distinct = sum(copies.values()), len(copies)
    repeats = sum(1 for it in sample["items"]
                  if copies[fingerprint(details[it['complaint_id']]['narrative'])] > 1)
    print(f"{n_distinct:,} distinct texts among {n_all:,} narratives. "
          f"{repeats} of {len(sample['items'])} sampled texts also appear in other complaints")
    return sample


def main(argv):
    force = argv == ["--force"]
    if argv and not force:
        sys.exit("usage: python -m eval.taxonomy_sample [--force]")
    try:
        run(force=force)
    except SampleError as err:
        sys.exit(f"ERROR {err}")


if __name__ == "__main__":
    main(sys.argv[1:])
