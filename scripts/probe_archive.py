"""Inspect CFPB narratives archive zips without extracting them.

Usage: python -I probe_archive.py <zip> [<zip> ...]

For each zip it reports
  1. What format is inside and how big it is
  2. What the columns are
  3. How many rows have a narrative, by month received and by product
  4. Whether rows carry the structured fields and a unique complaint ID
"""
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter

csv.field_size_limit(sys.maxsize)

# Column names seen in the live CFPB export, used only as a checklist.
EXPECTED = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company",
    "State", "ZIP code", "Tags", "Consumer consent provided?",
    "Submitted via", "Date sent to company", "Company response to consumer",
    "Timely response?", "Consumer disputed?", "Complaint ID",
]

# The archive page says it covers complaints received up to this date.
STATED_CUTOFF = "2026-08-14"


def mb(n):
    return f"{n / 1_048_576:,.1f} MB"


def pct(part, whole):
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def find(header, *words):
    for col in header:
        low = col.lower()
        if all(w in low for w in words):
            return col
    return None


def probe_csv(raw, name):
    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(text)
    header = reader.fieldnames or []
    print(f"\nColumns in {name} ({len(header)})")
    for col in header:
        print(f"  - {col}")
    missing = [c for c in EXPECTED if c not in header]
    extra = [c for c in header if c not in EXPECTED]
    print(f"\nExpected columns missing: {missing or 'none'}")
    print(f"Columns not in the live export: {extra or 'none'}")

    narr = find(header, "narrative")
    cid = find(header, "complaint", "id")
    date = find(header, "date", "received")
    prod = "Product" if "Product" in header else find(header, "product")

    rows = 0
    filled = Counter()
    ids = set()
    dup_ids = 0
    dates = []
    after_cutoff = 0
    month_rows, month_narr = Counter(), Counter()
    prod_rows, prod_narr = Counter(), Counter()
    lengths = []
    samples = []
    for row in reader:
        rows += 1
        for col in header:
            if (row.get(col) or "").strip():
                filled[col] += 1
        if cid:
            v = row.get(cid)
            if v in ids:
                dup_ids += 1
            ids.add(v)
        d = (row.get(date) or "") if date else ""
        if d:
            dates.append(d)
            if d > STATED_CUTOFF:
                after_cutoff += 1
        month = d[:7] or "unknown"
        p = (row.get(prod) or "") if prod else ""
        text_len = len((row.get(narr) or "").strip()) if narr else 0
        lengths.append(text_len)
        month_rows[month] += 1
        prod_rows[p] += 1
        if text_len:
            month_narr[month] += 1
            prod_narr[p] += 1
            if len(samples) < 3:
                samples.append(row)

    print(f"\nRows: {rows:,}")
    print("\nNon-empty count per column")
    for col in header:
        print(f"  {filled[col]:>10,}/{rows:,}  {col}")

    if cid:
        print(f"\nComplaint ID column: {cid!r}, duplicates: {dup_ids:,}")
    else:
        print("\nNo complaint ID column found")
    if dates:
        print(f"\nDate received range as text: {min(dates)} to {max(dates)}")
        print(f"Rows received after {STATED_CUTOFF}: {after_cutoff:,}")

    if not narr:
        print("\nNo narrative column found")
        return
    with_text = sorted(n for n in lengths if n)
    print(f"\nNarrative column: {narr!r}")
    print(f"  rows with a narrative: {len(with_text):,}/{rows:,} ({pct(len(with_text), rows).strip()})")
    if with_text:
        q = lambda f: with_text[min(len(with_text) - 1, int(f * len(with_text)))]
        print(f"  length of non-empty narratives in chars: median {q(0.5):,}, "
              f"90th pct {q(0.9):,}, max {with_text[-1]:,}")
        print(f"  total narrative text: {mb(sum(with_text))}")

    print("\nNarratives by month received")
    print("  month       rows   with narrative")
    for m in sorted(month_rows):
        print(f"  {m:<8} {month_rows[m]:>9,}   {month_narr[m]:>9,}  {pct(month_narr[m], month_rows[m])}")

    print("\nNarratives by product (top 12 by rows)")
    print("        rows   with narrative   product")
    for p, n in prod_rows.most_common(12):
        print(f"  {n:>9,}   {prod_narr[p]:>9,}  {pct(prod_narr[p], n)}   {p}")

    print("\nFirst narratives found, each cut to 300 chars")
    for s in samples:
        print("  ---")
        print(f"  {date}: {s.get(date)}   {prod}: {s.get(prod)}")
        print(f"  Issue: {s.get('Issue')}")
        print(f"  {(s.get(narr) or '')[:300]!r}")
    if not samples:
        print("  none")


def probe_json(raw, name):
    head = raw.read(4000).decode("utf-8", errors="replace")
    print(f"\nFirst 4000 bytes of {name}")
    print(head)
    first = head.lstrip()[:1]
    if first == "[":
        print("\nLooks like one JSON array. A streaming parser will be needed.")
    elif first == "{":
        line = head.splitlines()[0]
        try:
            obj = json.loads(line)
            print("\nLooks like JSON lines. Keys of the first record:")
            for k in obj:
                print(f"  - {k}")
        except json.JSONDecodeError:
            print("\nOne JSON object, not JSON lines.")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_zip(path):
    print("=" * 72)
    print(f"Zip: {path}")
    print(f"Bytes: {os.path.getsize(path):,}")
    print(f"SHA-256: {sha256_of(path)}")
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        print(f"Members: {len(members)}")
        for m in members:
            print(f"  {m.filename}  packed {mb(m.compress_size)}  unpacked {mb(m.file_size)}")
        for m in members:
            if m.is_dir():
                continue
            low = m.filename.lower()
            with zf.open(m) as raw:
                if low.endswith((".csv", ".txt", ".tsv")):
                    probe_csv(raw, m.filename)
                elif low.endswith((".json", ".jsonl", ".ndjson")):
                    probe_json(raw, m.filename)
                else:
                    print(f"\n{m.filename}: unknown type, first bytes {raw.read(200)!r}")
    print()


def main(paths):
    for path in paths:
        probe_zip(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -I probe_archive.py <zip> [<zip> ...]")
    main(sys.argv[1:])
