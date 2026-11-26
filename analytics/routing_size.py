"""Sizing for the deep-dive memo: fraud claims in narratives filed under non-fraud issues.

Usage: python -m analytics.routing_size

The 18,190 fraud complaints are counted from the issue the consumer picks, so
issue-based routing already finds them. This sizes what it misses: narratives
filed under an issue the fraud mapping does not count as fraud, whose story is
a fraud claim (the root-cause code fraud_claim).

Reads data/complaints.sqlite (read only), analytics/fraud_mapping.csv, the
test set eval/sets/test.json and the frozen tagger test log (which holds Prit's
label and the tagger's tag for each of the 200 test narratives). Writes
analytics/results/q7_routing_size.csv: counts only, no narrative text.

Per product:
- share: test narratives under a non-fraud issue that are labelled fraud_claim,
  with a Wilson 95% interval. The test set is 100 narratives per product drawn
  at random within each month, in proportion to the month's distinct texts
  (one narrative per fingerprint, the lowest complaint ID), January to July.
- base: the pool the test set was drawn from, rebuilt the same way (leaving out
  the fingerprints of the excluded sets) and counted under non-fraud issues.
  The rebuilt pool must reproduce the pool sizes saved in test.json month by
  month, or the script stops.
- estimate a month = share x base / months in the pool; its bounds come from
  the share's interval (the base is a count).
- the tagger: how many of those narratives it tagged fraud_claim, and how many
  of those tags match the label.
- for context: every narrative under a non-fraud issue, copies included,
  January to August.

The "Both products" row adds the two estimates; its bounds are the sums of the
two products' bounds, which is wider than a joint interval would be.
"""
import csv
import json
import os
import sqlite3
import sys

from analytics import mapping
from eval import labelset
from eval import taxonomy_sample as ts
from eval.tagrun import wilson
from ingest import sources
from ingest.textnorm import fingerprint

SET_PATH = "eval/sets/test.json"
LOG_PATH = "eval/runs/test-20261116T055317Z.jsonl"
OUT_PATH = "analytics/results/q7_routing_size.csv"
CODE = "fraud_claim"
BOTH = "Both products"
COLUMNS = ["product", "test_items", "test_fraud_issue", "test_fraud_issue_fraud_claim",
           "test_other_issue", "other_fraud_claim", "share", "share_low", "share_high",
           "other_tagged_fraud_claim", "other_tagged_right", "pool_other_issue",
           "pool_months", "est_per_month", "est_low", "est_high", "narratives_other_issue"]


class SizeError(Exception):
    pass


def load_log(path=LOG_PATH):
    """The test items of a tagger run log: complaint_id, product, label, tag, status."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = [r for r in rows if r.get("type") == "item" and r.get("set") == "test"]
    if not items:
        raise SizeError(f"{path}: no test items")
    return items


def is_fraud(fmap, product, issue, sub_issue):
    """The mapping's verdict for one complaint; a NULL sub-issue is blank in the mapping."""
    row = fmap.get((product, issue, sub_issue or ""))
    if row is None:
        raise SizeError(f"not in the fraud mapping: {product} / {issue} / {sub_issue}")
    return row["fraud"] == "yes"


def check_pool(pool_counts, allocation):
    """pool_counts {(product, month): n} must equal the sizes saved with the test set."""
    saved = {(a["product"], a["month"]): a["pool"] for a in allocation}
    if pool_counts != saved:
        diff = sorted(k for k in set(saved) | set(pool_counts)
                      if saved.get(k) != pool_counts.get(k))
        shown = ", ".join(f"{p} {m}: saved {saved.get((p, m))}, rebuilt {pool_counts.get((p, m))}"
                          for p, m in diff[:4])
        raise SizeError(f"the rebuilt pool doesn't match the test set's: {shown}")


def size(items, item_fraud, pool_other, pool_months, narr_other, products=sources.PRODUCTS):
    """The result rows.

    items: log items; item_fraud: {complaint_id: True if its issue is a fraud
    issue}; pool_other: {product: pool narratives under non-fraud issues};
    pool_months: {product: months in the pool}; narr_other: {product: all
    narratives under non-fraud issues}.
    """
    out = []
    for product in products:
        mine = [it for it in items if it["product"] == product]
        if not mine:
            raise SizeError(f"no test items for {product}")
        fraud_issue = [it for it in mine if item_fraud[it["complaint_id"]]]
        other = [it for it in mine if not item_fraud[it["complaint_id"]]]
        k = sum(it["label"] == CODE for it in other)
        n = len(other)
        lo, hi = wilson(k, n)
        share = k / n if n else 0.0
        tagged = [it for it in other if it["status"] == "tagged" and it["tag"] == CODE]
        months = pool_months[product]
        per_month = pool_other[product] / months
        out.append({
            "product": product,
            "test_items": len(mine),
            "test_fraud_issue": len(fraud_issue),
            "test_fraud_issue_fraud_claim": sum(it["label"] == CODE for it in fraud_issue),
            "test_other_issue": n,
            "other_fraud_claim": k,
            "share": round(share, 4),
            "share_low": round(lo, 4),
            "share_high": round(hi, 4),
            "other_tagged_fraud_claim": len(tagged),
            "other_tagged_right": sum(it["label"] == CODE for it in tagged),
            "pool_other_issue": pool_other[product],
            "pool_months": months,
            "est_per_month": round(share * per_month, 1),
            "est_low": round(lo * per_month, 1),
            "est_high": round(hi * per_month, 1),
            "narratives_other_issue": narr_other[product],
        })
    both = {c: "" for c in COLUMNS}
    both["product"] = BOTH
    for c in ("test_items", "test_fraud_issue", "test_fraud_issue_fraud_claim",
              "test_other_issue", "other_fraud_claim", "other_tagged_fraud_claim",
              "other_tagged_right", "pool_other_issue", "narratives_other_issue"):
        both[c] = sum(r[c] for r in out)
    for c in ("est_per_month", "est_low", "est_high"):
        both[c] = round(sum(r[c] for r in out), 1)
    return out + [both]


def _fetch_issues(conn, ids):
    found = {}
    ids = list(ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        marks = ", ".join("?" * len(chunk))
        for cid, product, issue, sub in conn.execute(
                f"SELECT complaint_id, product, issue, sub_issue FROM complaints "
                f"WHERE complaint_id IN ({marks})", chunk):
            found[cid] = (product, issue, sub)
    missing = [c for c in ids if c not in found]
    if missing:
        raise SizeError(f"IDs not in this database: {missing[:5]}")
    return found


def run(db_path=sources.DB_PATH, set_path=SET_PATH, log_path=LOG_PATH,
        mapping_path=mapping.MAPPING_PATH, out_path=OUT_PATH, out=print):
    fmap = mapping.load(mapping_path)
    with open(set_path, encoding="utf-8") as f:
        test = json.load(f)
    items = load_log(log_path)
    if sorted(it["complaint_id"] for it in items) != sorted(
            it["complaint_id"] for it in test["items"]):
        raise SizeError("the log's items are not the test set's items")
    conn = ts.connect_ro(db_path)
    try:
        # The pool, rebuilt as eval/labelset.py's create_set drew it.
        excluded_ids = [it["complaint_id"] for n in test["excluded_sets"]
                        for it in labelset.load_set(n)["items"]]
        excluded_fps = {fingerprint(t) for t in labelset.texts_for(conn, excluded_ids).values()}
        pool, _ = ts.scan(conn)
        texts = labelset.texts_for(conn, [cid for ids in pool.values() for cid in ids])
        pool = {k: [cid for cid in ids if fingerprint(texts[cid]) not in excluded_fps]
                for k, ids in pool.items()}
        pool = {k: ids for k, ids in pool.items() if ids}
        check_pool({k: len(ids) for k, ids in pool.items()}, test["allocation"])

        pool_ids = [cid for ids in pool.values() for cid in ids]
        issues = _fetch_issues(conn, pool_ids + [it["complaint_id"] for it in items])
        item_fraud = {it["complaint_id"]: is_fraud(fmap, *issues[it["complaint_id"]])
                      for it in items}
        pool_other = {p: 0 for p in sources.PRODUCTS}
        for cid in pool_ids:
            product = issues[cid][0]
            if not is_fraud(fmap, *issues[cid]):
                pool_other[product] += 1
        pool_months = {p: sum(1 for (q, _m) in pool if q == p) for p in sources.PRODUCTS}

        narr_other = {p: 0 for p in sources.PRODUCTS}
        for product, issue, sub, n in conn.execute(
                "SELECT product, issue, sub_issue, COUNT(*) FROM complaints "
                "WHERE has_narrative = 1 GROUP BY product, issue, sub_issue"):
            if not is_fraud(fmap, product, issue, sub):
                narr_other[product] += n
    finally:
        conn.close()

    rows = size(items, item_fraud, pool_other, pool_months, narr_other)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["product"] == BOTH:
            out(f"{BOTH}: about {r['est_per_month']} a month "
                f"({r['est_low']} to {r['est_high']}, sum of the products' bounds)")
        else:
            out(f"{r['product']}: {r['other_fraud_claim']} of {r['test_other_issue']} test "
                f"narratives under non-fraud issues are fraud_claim "
                f"({r['share']:.1%}, {r['share_low']:.1%} to {r['share_high']:.1%}); "
                f"pool {r['pool_other_issue']} over {r['pool_months']} months; "
                f"about {r['est_per_month']} a month ({r['est_low']} to {r['est_high']})")
    out(f"Written: {out_path}")
    return rows


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        run()
    except (SizeError, mapping.MappingError, ts.SampleError, labelset.LabelError,
            OSError, sqlite3.Error) as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
