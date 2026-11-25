"""Load and check the fraud mapping (analytics/fraud_mapping.csv).

Usage: python -m analytics.mapping     check the mapping against issue_values.csv

One row per (product, issue, sub_issue) value in analytics/issue_values.csv,
each exactly once (decisions 55 and 59):
  fraud       yes or no
  kind        unauthorized_transaction or new_account when fraud is yes, else empty
  borderline  yes or no: a row whose category can't tell fraud from other causes
  reason      one line, never empty
A blank sub_issue stands for NULL in the data. The mapping was written from the
values and committed before any analysis query was run.
"""
import csv
import sys

MAPPING_PATH = "analytics/fraud_mapping.csv"
VALUES_PATH = "analytics/issue_values.csv"
COLUMNS = ["product", "issue", "sub_issue", "fraud", "kind", "borderline", "reason"]
KINDS = {"unauthorized_transaction", "new_account"}


class MappingError(Exception):
    pass


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def load(path=MAPPING_PATH):
    """{(product, issue, sub_issue): row}, or MappingError listing every problem."""
    cols, rows = _read(path)
    if cols != COLUMNS:
        raise MappingError(f"{path}: columns must be {COLUMNS}, found {cols}")
    problems, out = [], {}
    for i, r in enumerate(rows, 2):
        key = (r["product"], r["issue"], r["sub_issue"])
        where = f"line {i} ({' / '.join(k for k in key if k)})"
        if key in out:
            problems.append(f"{where}: appears twice")
        if r["fraud"] not in ("yes", "no"):
            problems.append(f"{where}: fraud must be yes or no")
        if r["borderline"] not in ("yes", "no"):
            problems.append(f"{where}: borderline must be yes or no")
        if r["fraud"] == "yes" and r["kind"] not in KINDS:
            problems.append(f"{where}: a fraud row needs a kind from {sorted(KINDS)}")
        if r["fraud"] == "no" and r["kind"]:
            problems.append(f"{where}: kind must be empty when fraud is no")
        if not r["reason"].strip():
            problems.append(f"{where}: reason is empty")
        out[key] = r
    if problems:
        raise MappingError(f"{path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return out


def check_coverage(mapping, values_path=VALUES_PATH):
    """Every value row is mapped and every mapped row is a value row."""
    _, values = _read(values_path)
    keys = {(r["product"], r["issue"], r["sub_issue"]) for r in values}
    missing = sorted(keys - set(mapping))
    extra = sorted(set(mapping) - keys)
    if missing or extra:
        raise MappingError(f"not mapped: {missing[:3]}{'...' if len(missing) > 3 else ''}; "
                           f"mapped but not in the data: {extra[:3]}")
    counts = {(r["product"], r["issue"], r["sub_issue"]): int(r["complaints"]) for r in values}
    return counts


def main(argv):
    try:
        m = load()
        counts = check_coverage(m)
    except (MappingError, OSError) as e:
        print(f"ERROR {e}")
        return 1
    total = sum(counts.values())
    fraud = sum(n for k, n in counts.items() if m[k]["fraud"] == "yes")
    print(f"{len(m)} rows, all values mapped once; fraud {fraud} of {total} complaints "
          f"({100 * fraud / total:.1f}%)")
    for kind in sorted(KINDS):
        n = sum(c for k, c in counts.items() if m[k]["kind"] == kind)
        print(f"  {kind}: {n}")
    border = sum(n for k, n in counts.items() if m[k]["borderline"] == "yes" and m[k]["fraud"] == "no")
    print(f"  borderline rows left out: {border} complaints")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
