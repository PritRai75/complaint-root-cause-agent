"""Exports from data/complaints.sqlite for the fraud and scam trend investigation.

Usage:
  python -m analytics.export values    the issue and sub-issue values with counts
  python -m analytics.export table     the complaints table as a CSV for BigQuery

values writes analytics/issue_values.csv: one row per (product, issue,
sub_issue) with its complaints and narratives counts, January to August 2026.
Counts only, no narrative text, so it is committed. The fraud and scam mapping
(analytics/fraud_mapping.csv) is written from it before any analysis.

table writes build/analytics/complaints.csv (gitignored with the rest of
build/): every column of the complaints table, header first, UTF-8, LF line
ends, NULL as an empty field (BigQuery's default null marker). It refuses a
file over 100 MB, the BigQuery console's limit for a local upload, and writes
build/analytics/complaints.manifest.json with the row count, bytes, SHA-256 and
the database's build date. The narratives table is never exported. The
upload's schema is analytics/schema.json.

The database is opened read only.
"""
import csv
import hashlib
import json
import os
import sqlite3
import sys

from ingest.build_db import DB_COLUMNS

DB_PATH = "data/complaints.sqlite"
VALUES_PATH = "analytics/issue_values.csv"
TABLE_DIR = "build/analytics"
CONSOLE_LIMIT = 100 * 1000 * 1000


class ExportError(Exception):
    pass


def connect(db_path):
    if not os.path.exists(db_path):
        raise ExportError(f"{db_path} not found: build it with python -m ingest.build_db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(complaints)")]
    if cols != DB_COLUMNS:
        conn.close()
        raise ExportError(f"complaints columns differ from the loader's: {cols}")
    return conn


def export_values(db_path=DB_PATH, out_path=VALUES_PATH, out=print):
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT product, issue, COALESCE(sub_issue, ''), COUNT(*), SUM(has_narrative) "
            "FROM complaints GROUP BY product, issue, sub_issue "
            "ORDER BY product, COUNT(*) DESC, issue, COALESCE(sub_issue, '')").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    finally:
        conn.close()
    if sum(r[3] for r in rows) != total:
        raise ExportError("value counts don't add up to the table")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["product", "issue", "sub_issue", "complaints", "narratives"])
        w.writerows(rows)
    out(f"{len(rows)} product-issue-sub-issue rows, {total} complaints")
    for p in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == p]
        out(f"  {p}: {len(sub)} rows, {len({r[1] for r in sub})} issues, "
            f"{sum(r[3] for r in sub)} complaints")
    out(f"Written: {out_path}")
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_table(db_path=DB_PATH, out_dir=TABLE_DIR, limit=CONSOLE_LIMIT, out=print):
    conn = connect(db_path)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "complaints.csv")
    try:
        built = dict(conn.execute("SELECT key, value FROM build_info").fetchall())
        n = 0
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(DB_COLUMNS)
            for r in conn.execute(f"SELECT {', '.join(DB_COLUMNS)} FROM complaints "
                                  "ORDER BY complaint_id"):
                w.writerow(["" if v is None else v for v in r])
                n += 1
    finally:
        conn.close()
    size = os.path.getsize(path)
    if size > limit:
        raise ExportError(f"{path} is {size:,} bytes, over the console's "
                          f"{limit:,}-byte upload limit")
    manifest = {"file": "complaints.csv", "rows": n, "bytes": size,
                "sha256": sha256_file(path), "columns": DB_COLUMNS,
                "db_built_at": built.get("built_at")}
    with open(os.path.join(out_dir, "complaints.manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    out(f"{n} rows, {size / 1e6:.1f} MB, SHA-256 {manifest['sha256'][:12]}")
    out(f"Written: {path}")
    return manifest


def main(argv):
    if len(argv) != 2 or argv[1] not in ("values", "table"):
        print(__doc__)
        return 2
    try:
        (export_values if argv[1] == "values" else export_table)()
    except (ExportError, OSError, sqlite3.Error) as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
