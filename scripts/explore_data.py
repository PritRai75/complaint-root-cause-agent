"""First look at the built database, to settle the semantic layer's definitions.

Usage (from the repo root, output to a file you can attach):
  PYTHONUTF8=1 python -m scripts.explore_data > scratch/explore.txt

Read-only. Prints value lists for the response fields, complaints still in
progress by month, top issues and sub-issues per product with narrative
coverage, coverage by submission channel, narrative length, template
duplicates (exact after normalising, so a floor), text that looks like
instructions to an AI, and how concentrated complaints are by company.
"""
import os
import sqlite3
import sys
from collections import Counter, defaultdict

from ingest import sources
from ingest.textnorm import fingerprint

AI_PATTERNS = ("ignore previous", "ignore all previous", "ignore the above",
               "system prompt", "chatgpt", "as an ai", "language model")


def connect_ro(db_path):
    if not os.path.exists(db_path):
        sys.exit(f"ERROR {db_path} not found. Run python -m ingest.build_db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def pct(part, whole):
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def table(title, header, rows):
    """Plain-text table. Integers are right-aligned with thousands separators,
    everything else is left-aligned. Pass IDs as strings so they keep no commas."""
    right = [all(isinstance(r[i], int) or r[i] is None for r in rows) and bool(rows)
             for i in range(len(header))]
    cells = [[("" if v is None else f"{v:,}" if isinstance(v, int) else str(v)) for v in r]
             for r in rows]
    widths = [max([len(str(h))] + [len(r[i]) for r in cells]) for i, h in enumerate(header)]

    def fmt(row):
        return "  ".join((c.rjust(w) if right[i] else c.ljust(w)).rstrip() if i == len(row) - 1
                         else (c.rjust(w) if right[i] else c.ljust(w))
                         for i, (c, w) in enumerate(zip(row, widths)))

    lines = ["", f"== {title}", ""]
    lines.append(fmt([str(h) for h in header]))
    lines.append("  ".join("-" * w for w in widths))
    lines += [fmt(r) for r in cells] or ["(none)"]
    return lines


def value_counts(conn, column):
    rows = conn.execute(
        f"SELECT COALESCE({column}, '(blank)'), "
        "SUM(product = 'Credit card'), SUM(product = 'Checking or savings account'), COUNT(*) "
        f"FROM complaints GROUP BY 1 ORDER BY 4 DESC").fetchall()
    return table(f"{column} by product", [column, "card", "checking", "total"], rows)


def report(conn):
    out = []
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    out += ["FIRST LOOK AT data/complaints.sqlite",
            f"built {info.get('built_at')}  loader {info.get('loader_commit')}  "
            f"complaints {int(info['complaints_rows']):,}  narratives {int(info['narratives_rows']):,}"]

    for col in ("company_response", "timely_response", "company_public_response",
                "submitted_via", "tags"):
        out += value_counts(conn, col)

    rows = conn.execute(
        "SELECT substr(date_received, 1, 7) AS m, COUNT(*), "
        "COUNT(CASE WHEN company_response = 'In progress' THEN 1 END), "
        "COUNT(CASE WHEN company_response IS NULL THEN 1 END), "
        "COUNT(CASE WHEN timely_response = 'Yes' THEN 1 END), "
        "COUNT(CASE WHEN timely_response = 'No' THEN 1 END), "
        "COUNT(CASE WHEN timely_response IS NULL THEN 1 END) "
        "FROM complaints GROUP BY m ORDER BY m").fetchall()
    out += table("Open complaints and timely response by month received",
                 ["month", "complaints", "in progress", "no response", "timely yes",
                  "timely no", "timely blank"], rows)

    rows = conn.execute(
        "SELECT COALESCE(company_response, '(blank)'), COALESCE(timely_response, '(blank)'), "
        "COUNT(*) FROM complaints GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    out += table("Company response against timely response", ["company_response", "timely", "n"],
                 rows)

    rows = conn.execute(
        "SELECT CAST(julianday(date_sent_to_company) - julianday(date_received) AS INTEGER) AS d, "
        "COUNT(*) FROM complaints GROUP BY d ORDER BY d").fetchall()
    head = [r for r in rows if r[0] is None or r[0] <= 7]
    tail = sum(r[1] for r in rows if r[0] is not None and r[0] > 7)
    out += table("Days from received to sent to company", ["days", "complaints"],
                 [["(blank)" if d is None else d, n] for d, n in head] + [["more than 7", tail]])

    for product in sources.PRODUCTS:
        rows = conn.execute(
            "SELECT COALESCE(sub_product, '(blank)'), COUNT(*), SUM(has_narrative) "
            "FROM complaints WHERE product = ? GROUP BY 1 ORDER BY 2 DESC", (product,)).fetchall()
        out += table(f"Sub-products, {product}", ["sub_product", "complaints", "narratives", "coverage"],
                     [[s, n, k, pct(k, n)] for s, n, k in rows])

        total = conn.execute("SELECT COUNT(*), SUM(has_narrative) FROM complaints "
                             "WHERE product = ?", (product,)).fetchone()
        rows = conn.execute(
            "SELECT issue, COUNT(*), SUM(has_narrative) FROM complaints WHERE product = ? "
            "GROUP BY issue ORDER BY 2 DESC", (product,)).fetchall()
        out += table(f"Issues, {product} (all {len(rows)})",
                     ["issue", "complaints", "share", "narratives", "share of narr.", "coverage"],
                     [[i, n, pct(n, total[0]), k, pct(k, total[1]), pct(k, n)] for i, n, k in rows])

        rows = conn.execute(
            "SELECT issue || '  >  ' || COALESCE(sub_issue, '(blank)'), COUNT(*), SUM(has_narrative) "
            "FROM complaints WHERE product = ? GROUP BY issue, sub_issue ORDER BY 2 DESC LIMIT 20",
            (product,)).fetchall()
        out += table(f"Top 20 issue > sub-issue pairs, {product}",
                     ["issue > sub_issue", "complaints", "narratives"], rows)

    rows = conn.execute(
        "SELECT COALESCE(submitted_via, '(blank)'), COUNT(*), SUM(has_narrative) "
        "FROM complaints GROUP BY 1 ORDER BY 2 DESC").fetchall()
    out += table("Narrative coverage by submission channel",
                 ["submitted_via", "complaints", "narratives", "coverage"],
                 [[v, n, k, pct(k, n)] for v, n, k in rows])

    lengths = defaultdict(list)
    fps = Counter()
    fp_meta = defaultdict(lambda: {"products": Counter(), "months": set(), "example": None})
    ai_hits = Counter()
    ai_ids = defaultdict(list)
    for cid, product, month, text, n in conn.execute(
            "SELECT c.complaint_id, c.product, substr(c.date_received, 1, 7), n.narrative, "
            "n.char_length FROM narratives n JOIN complaints c USING (complaint_id) "
            "ORDER BY c.complaint_id"):
        lengths[product].append(n)
        fp = fingerprint(text)
        fps[fp] += 1
        meta = fp_meta[fp]
        meta["products"][product] += 1
        meta["months"].add(month)
        if meta["example"] is None:
            meta["example"] = (cid, " ".join(text.split())[:160])
        low = text.lower()
        for p in AI_PATTERNS:
            if p in low:
                ai_hits[p] += 1
                if len(ai_ids[p]) < 5:
                    ai_ids[p].append(cid)

    def q(values, p):
        return values[min(len(values) - 1, int(p * len(values)))]

    rows = []
    for product in sources.PRODUCTS:
        v = sorted(lengths[product])
        if v:
            rows.append([product, len(v), q(v, .1), q(v, .5), q(v, .9), q(v, .99), v[-1],
                         sum(1 for x in v if x > 4000)])
    out += table("Narrative length in characters",
                 ["product", "narratives", "p10", "median", "p90", "p99", "max", "over 4,000"], rows)

    groups = {fp: n for fp, n in fps.items() if n > 1}
    in_groups = sum(groups.values())
    total_n = sum(fps.values())
    out += ["", "== Template duplicates (same text after normalising redactions, amounts, "
            "numbers, case and punctuation)", "",
            f"{len(groups):,} groups of 2 or more, holding {in_groups:,} of {total_n:,} narratives "
            f"({pct(in_groups, total_n).strip()}). Exact match only, so this is a floor.",
            f"Narratives that would remain after keeping one per group: "
            f"{total_n - in_groups + len(groups):,}"]
    sizes = Counter(min(n, 10) for n in groups.values())
    out += table("Group sizes", ["copies", "groups"],
                 [["10 or more" if s == 10 else s, sizes[s]] for s in sorted(sizes)])
    rows = []
    for fp, n in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        m = fp_meta[fp]
        cid, text = m["example"]
        prods = "/".join("card" if p == "Credit card" else "checking" for p in sorted(m["products"]))
        rows.append([n, prods, f"{min(m['months'])}..{max(m['months'])}", str(cid), text])
    out += table("Largest 15 groups", ["copies", "products", "months", "example id",
                                       "start of text"], rows)

    out += table("Narratives containing AI-instruction-like phrases",
                 ["phrase", "narratives", "first ids"],
                 [[p, ai_hits[p], " ".join(map(str, ai_ids[p]))] for p in AI_PATTERNS])

    n_companies = conn.execute("SELECT COUNT(DISTINCT company) FROM complaints").fetchone()[0]
    rows = conn.execute(
        "SELECT company, COUNT(*), SUM(has_narrative) FROM complaints "
        "GROUP BY company ORDER BY 2 DESC LIMIT 10").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    out += table(f"Company concentration ({n_companies:,} companies; top 10 shown)",
                 ["company", "complaints", "share", "narratives", "coverage"],
                 [[c, n, pct(n, total), k, pct(k, n)] for c, n, k in rows])
    top10 = sum(r[1] for r in rows)
    out += [f"Top 10 companies hold {pct(top10, total).strip()} of complaints."]
    return out


def main(argv):
    if argv:
        sys.exit("usage: python -m scripts.explore_data")
    conn = connect_ro(sources.DB_PATH)
    try:
        print("\n".join(report(conn)))
    finally:
        conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
