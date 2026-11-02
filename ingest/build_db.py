"""Build data/complaints.sqlite from the archive zips.

Usage: python -m ingest.build_db [--force]

Reads each zip in sources.FILES straight from disk (nothing is extracted),
checks it against data/manifest.json, keeps the products and dates set in
sources.py, and writes two tables:

  complaints  one row per complaint, structured fields only
  narratives  complaint_id and the narrative text, for rows that have one

Narrative text is stored exactly as published, redaction markers included.
The ZIP code column is dropped. A build_info table records what was built
and from which files.
"""
import csv
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone

from ingest import sources
from ingest.fetch import load_manifest, sha256_of

csv.field_size_limit(sys.maxsize)

# CSV column -> database column. Every CSV column must appear here,
# in DROPPED or as NARRATIVE, so a change in the export stops the build.
COLUMN_MAP = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Company public response": "company_public_response",
    "Timely response?": "timely_response",
}
NARRATIVE = "Consumer complaint narrative"
DROPPED = {"ZIP code"}
DATE_COLUMNS = ("Date received", "Date sent to company")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BATCH = 5000

DB_COLUMNS = list(COLUMN_MAP.values()) + ["has_narrative", "source_file"]

SCHEMA = """
CREATE TABLE complaints (
    complaint_id            INTEGER PRIMARY KEY,
    date_received           TEXT NOT NULL,
    product                 TEXT NOT NULL,
    sub_product             TEXT,
    issue                   TEXT NOT NULL,
    sub_issue               TEXT,
    company                 TEXT NOT NULL,
    state                   TEXT,
    tags                    TEXT,
    submitted_via           TEXT,
    date_sent_to_company    TEXT,
    company_response        TEXT,
    company_public_response TEXT,
    timely_response         TEXT,
    has_narrative           INTEGER NOT NULL CHECK (has_narrative IN (0, 1)),
    source_file             TEXT NOT NULL
);
CREATE TABLE narratives (
    complaint_id INTEGER PRIMARY KEY REFERENCES complaints (complaint_id),
    narrative    TEXT NOT NULL,
    char_length  INTEGER NOT NULL
);
CREATE TABLE build_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

INDEXES = """
CREATE INDEX idx_complaints_date    ON complaints (date_received);
CREATE INDEX idx_complaints_product ON complaints (product);
CREATE INDEX idx_complaints_issue   ON complaints (issue);
CREATE INDEX idx_complaints_company ON complaints (company);
"""


class BuildError(Exception):
    pass


def check_header(header, name):
    known = set(COLUMN_MAP) | DROPPED | {NARRATIVE}
    missing = sorted(known - set(header))
    unknown = sorted(set(header) - known)
    if missing or unknown:
        raise BuildError(f"{name}: columns changed. Missing {missing}, unknown {unknown}")


def to_records(row, source_file, products, date_from, date_to):
    """Return (complaint, narrative or None) for a kept row, or None.

    Raises BuildError on values that break our assumptions, so a format
    change in the archive stops the build instead of loading bad data.
    """
    for col in DATE_COLUMNS:
        value = row[col]
        if value and not ISO_DATE.match(value):
            raise BuildError(f"{source_file}: {col} not in YYYY-MM-DD form: {value!r}")
    if row["Product"] not in products:
        return None
    if not (date_from <= row["Date received"] <= date_to):
        return None

    cid = row["Complaint ID"]
    if not cid.isdigit():
        raise BuildError(f"{source_file}: complaint ID is not a number: {cid!r}")
    if row["Timely response?"] not in ("Yes", "No", ""):
        raise BuildError(f"{source_file}: unexpected timely response {row['Timely response?']!r}")

    text = row[NARRATIVE]
    has_text = bool(text.strip())
    values = []
    for csv_col, db_col in COLUMN_MAP.items():
        v = row[csv_col]
        if db_col == "complaint_id":
            values.append(int(v))
        else:
            values.append(v if v != "" else None)
    values += [int(has_text), source_file]
    narrative = (int(cid), text, len(text)) if has_text else None
    return tuple(values), narrative


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "unknown"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def verify_against_manifest(files, raw_dir, manifest_path):
    manifest = load_manifest(manifest_path)
    entries = []
    for name in files:
        path = os.path.join(raw_dir, name)
        known = manifest["files"].get(name)
        if not os.path.exists(path):
            raise BuildError(f"{name}: not found in {raw_dir}. Run python -m ingest.fetch")
        if known is None:
            raise BuildError(f"{name}: not in {manifest_path}. Run python -m ingest.fetch")
        digest = sha256_of(path)
        if digest != known["sha256"]:
            raise BuildError(f"{name}: SHA-256 differs from the manifest")
        entries.append({"file": name, "sha256": digest, "bytes": known["bytes"]})
    return entries


def read_zip(path):
    """Yield the rows of the single CSV inside a zip."""
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise BuildError(f"{path}: expected one CSV, found {[m.filename for m in members]}")
        with zf.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)
            check_header(reader.fieldnames or [], os.path.basename(path))
            yield from reader


def build(files=None, raw_dir=sources.RAW_DIR, manifest_path=sources.MANIFEST,
          db_path=sources.DB_PATH, products=sources.PRODUCTS,
          date_from=sources.DATE_FROM, date_to=sources.DATE_TO, force=False):
    files = files if files is not None else sources.FILES
    if os.path.exists(db_path) and not force:
        raise BuildError(f"{db_path} exists. Use --force to rebuild it")

    print("Checking files against the manifest")
    entries = verify_against_manifest(files, raw_dir, manifest_path)

    tmp_path = db_path + ".building"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executescript(SCHEMA)
        insert_c = (f"INSERT INTO complaints ({', '.join(DB_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(DB_COLUMNS))})")
        insert_n = "INSERT INTO narratives (complaint_id, narrative, char_length) VALUES (?, ?, ?)"

        replaced = 0
        for entry in entries:
            name = entry["file"]
            scanned = kept = with_text = 0
            batch_c, batch_n = [], []
            for row in read_zip(os.path.join(raw_dir, name)):
                scanned += 1
                result = to_records(row, name, products, date_from, date_to)
                if result is None:
                    continue
                complaint, narrative = result
                kept += 1
                replaced += sum(str(v).count("�") for v in complaint if v is not None)
                batch_c.append(complaint)
                if narrative:
                    with_text += 1
                    replaced += narrative[1].count("�")
                    batch_n.append(narrative)
                if len(batch_c) >= BATCH:
                    conn.executemany(insert_c, batch_c)
                    conn.executemany(insert_n, batch_n)
                    batch_c, batch_n = [], []
            conn.executemany(insert_c, batch_c)
            conn.executemany(insert_n, batch_n)
            conn.commit()
            entry.update(rows_scanned=scanned, rows_kept=kept, narratives_kept=with_text)
            print(f"  {name}: scanned {scanned:,}, kept {kept:,}, with narrative {with_text:,}")

        conn.executescript(INDEXES)
        n_complaints = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        n_narratives = conn.execute("SELECT COUNT(*) FROM narratives").fetchone()[0]
        flagged = conn.execute("SELECT COUNT(*) FROM complaints WHERE has_narrative = 1").fetchone()[0]
        orphans = conn.execute(
            "SELECT COUNT(*) FROM narratives n LEFT JOIN complaints c USING (complaint_id) "
            "WHERE c.complaint_id IS NULL").fetchone()[0]
        if flagged != n_narratives or orphans:
            raise BuildError(f"narrative counts disagree: flagged {flagged}, "
                             f"narratives {n_narratives}, orphans {orphans}")

        info = {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "loader_commit": git_commit(),
            "archive_page": sources.ARCHIVE_PAGE,
            "products": json.dumps(list(products)),
            "date_from": date_from,
            "date_to": date_to,
            "source_files": json.dumps(entries, indent=1),
            "complaints_rows": str(n_complaints),
            "narratives_rows": str(n_narratives),
            "replacement_chars": str(replaced),
        }
        conn.executemany("INSERT INTO build_info (key, value) VALUES (?, ?)", info.items())
        conn.commit()
        conn.execute("ANALYZE")
        conn.execute("VACUUM")
        months = conn.execute(
            "SELECT substr(date_received, 1, 7) AS month, COUNT(*), SUM(has_narrative) "
            "FROM complaints GROUP BY month ORDER BY month").fetchall()
    except BaseException:
        conn.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    conn.close()
    os.replace(tmp_path, db_path)

    print(f"\nWrote {db_path}  {os.path.getsize(db_path) / 1_048_576:,.1f} MB")
    print(f"complaints {n_complaints:,}   narratives {n_narratives:,}   "
          f"replacement characters {replaced:,}")
    print("\nmonth      complaints  with narrative")
    for month, total, narr in months:
        print(f"{month}  {total:>11,}  {narr:>9,}  {100 * narr / total:5.1f}%")
    return info


def main(argv):
    force = argv == ["--force"]
    if argv and not force:
        sys.exit("usage: python -m ingest.build_db [--force]")
    try:
        build(force=force)
    except BuildError as e:
        sys.exit(f"ERROR {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
