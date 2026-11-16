"""Error analysis of a tagger run and the comparison with the CFPB issue column.

Usage:
  python -m eval.taganalysis eval/runs/<run_id>.jsonl

Reads a run log written by eval.tagrun, Prit's labels and data/complaints.sqlite
(read only), prints the analysis and writes eval/runs/analysis-<run_id>.json.
For the test split it takes only the --final log. A dev log is accepted for
checking. Rules fixed before the tagger's test run (session 10):

Runner-up matches (decision 45). A label note that starts "vs <code>" names
Prit's runner-up. A miss (tag differs from the label, untagged items included)
"matches the runner-up" only when the tagger's code equals that one code. This
describes the misses. It is not a score: exact agreement stays the only
accuracy figure (decision 20).

Issue column (decision 46). The 200 labels are cross-tabulated by the issue
the consumer picked, grouped by product and issue (an issue name can appear
under both products). The issue-only bound is the best any lookup from
(product, issue) to one root cause could do: the most common label in each
group, summed. It is fitted on the same items it is scored on, so it is an
upper bound, shown beside the tagger's exact agreement on the same items.
"""
import hashlib
import json
import os
import re
import sqlite3
import sys
import tomllib
from collections import Counter

from eval import taxonomy
from eval import tagrun

DB_PATH = "data/complaints.sqlite"
VS = re.compile(r"^vs ([a-z_]+)\b")


class AnalysisError(Exception):
    pass


def read_log(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or rows[0].get("type") != "header":
        raise AnalysisError(f"{path} has no header line: is it an eval.tagrun log?")
    if rows[-1].get("type") != "summary":
        raise AnalysisError(f"{path} has no summary line: the run didn't finish")
    header, summary = rows[0], rows[-1]
    items = [r for r in rows if r.get("type") == "item"]
    if header.get("set") not in tagrun.SETS:
        raise AnalysisError(f"unknown set {header.get('set')!r} in the log")
    if header["set"] == "test" and not header.get("final"):
        raise AnalysisError("a test log must come from the --final run")
    if header.get("limit"):
        raise AnalysisError("the log is from a --limit run: analyse a full run")
    if len(items) != header.get("items"):
        raise AnalysisError(f"the header says {header.get('items')} items, the log has "
                            f"{len(items)}")
    return header, items, summary


def load_notes(kind, labels_dir=tagrun.LABELS_DIR):
    """{ref: (complaint_id, label, note)} from the label files of a set."""
    notes = {}
    for name in tagrun.SETS[kind]:
        with open(os.path.join(labels_dir, f"{name}.toml"), "rb") as f:
            labels = tomllib.load(f)
        for it in labels["item"]:
            notes[it["ref"]] = (it["complaint_id"], it["code"], it.get("note", ""))
    return notes


def runner_up(note, codes):
    """The code named by a note starting 'vs <code>', or None."""
    m = VS.match(note or "")
    if m and m.group(1) in codes:
        return m.group(1)
    return None


def check_labels(items, notes):
    """The labels must be the ones the run was scored against."""
    if set(notes) != {it["ref"] for it in items}:
        raise AnalysisError("the label files and the log hold different items")
    for it in items:
        cid, label, _ = notes[it["ref"]]
        if cid != it["complaint_id"] or label != it["label"]:
            raise AnalysisError(f"{it['ref']}: the label file has changed since the run "
                                f"({it['label']} in the log, {label} now)")


def misses_by_runner_up(items, notes, codes):
    scored = [it for it in items if not it["uncertain"]]
    out = {"misses": 0, "with_runner_up_note": 0, "match_runner_up": 0,
           "by_product": {}, "by_label": {}, "items": []}

    def bump(group, key, field):
        g = out[group].setdefault(key, {"misses": 0, "match_runner_up": 0})
        g[field] += 1

    for it in scored:
        tag = it["tag"] if it["status"] == "tagged" else None
        if tag == it["label"]:
            continue
        ru = runner_up(notes[it["ref"]][2], codes)
        match = tag is not None and tag == ru
        out["misses"] += 1
        out["with_runner_up_note"] += ru is not None
        out["match_runner_up"] += match
        for group, key in (("by_product", it["product"]), ("by_label", it["label"])):
            bump(group, key, "misses")
            if match:
                bump(group, key, "match_runner_up")
        out["items"].append({"ref": it["ref"], "complaint_id": it["complaint_id"],
                             "product": it["product"], "label": it["label"],
                             "tag": tag or it["status"], "runner_up": ru, "match": match})
    for group in ("by_product", "by_label"):
        out[group] = dict(sorted(out[group].items()))
    return out


def fetch_issues(ids, db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise AnalysisError(f"{db_path} not found: build it with python -m ingest.build_db")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        found = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = ("SELECT complaint_id, product, issue, sub_issue FROM complaints "
                 f"WHERE complaint_id IN ({','.join('?' * len(chunk))})")
            for cid, product, issue, sub in conn.execute(q, chunk):
                found[cid] = (product, issue, sub)
    finally:
        conn.close()
    missing = sorted(set(ids) - set(found))
    if missing:
        raise AnalysisError(f"{len(missing)} complaint IDs not in {db_path}, e.g. {missing[0]}")
    return found


def issue_table(items, issues):
    scored = [it for it in items if not it["uncertain"]]
    for it in scored:
        if issues[it["complaint_id"]][0] != it["product"]:
            raise AnalysisError(f"{it['ref']}: product differs between the labels and "
                                f"the database")
    groups = {}
    for it in scored:
        key = (it["product"], issues[it["complaint_id"]][1])
        groups.setdefault(key, []).append(it)
    rows = []
    for (product, issue), members in groups.items():
        labels = Counter(it["label"] for it in members)
        top = max(labels.values())
        rows.append({"product": product, "issue": issue, "n": len(members),
                     "causes": len(labels), "majority": top,
                     "majority_labels": sorted(c for c, k in labels.items() if k == top),
                     "labels": dict(labels.most_common()),
                     "tagger_agree": sum(it["status"] == "tagged" and it["tag"] == it["label"]
                                         for it in members)})
    rows.sort(key=lambda r: (-r["n"], r["product"], r["issue"]))
    n = len(scored)
    bound = sum(r["majority"] for r in rows)
    agree = sum(r["tagger_agree"] for r in rows)
    mixed = [r for r in rows if r["causes"] >= 2]
    return {"scored": n, "groups": len(rows),
            "groups_with_2plus_causes": len(mixed),
            "items_in_those_groups": sum(r["n"] for r in mixed),
            "issue_only_bound": tagrun_rate(bound, n),
            "tagger_exact": tagrun_rate(agree, n), "rows": rows}


def tagrun_rate(k, n):
    lo, hi = tagrun.wilson(k, n)
    return {"k": k, "n": n, "wilson95": [round(lo, 3), round(hi, 3)]}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def analyse(log_path, labels_dir=tagrun.LABELS_DIR, db_path=DB_PATH, taxonomy_path=None,
            runs_dir=tagrun.RUNS_DIR, out=print):
    header, items, summary = read_log(log_path)
    tax = taxonomy.load(taxonomy_path)
    if header["taxonomy_version"] != tax["version"]:
        raise AnalysisError(f"the run used taxonomy {header['taxonomy_version']}, the "
                            f"taxonomy is now {tax['version']}")
    codes = {c["code"] for c in tax["cause"]}
    notes = load_notes(header["set"], labels_dir)
    check_labels(items, notes)
    s = tagrun.score(items)
    if s["agree"] != summary["agree"] or s["scored"] != summary["scored"]:
        raise AnalysisError("re-scoring the log doesn't give its own summary")

    ru = misses_by_runner_up(items, notes, codes)
    issues = fetch_issues([it["complaint_id"] for it in items], db_path)
    table = issue_table(items, issues)

    result = {"type": "tag_analysis", "run_id": header["run_id"], "set": header["set"],
              "final": header.get("final", False), "log_sha256": sha256_file(log_path),
              "run_commit": header.get("code_commit"), "analysis_commit": tagrun.git_commit(),
              "prompt_version": header.get("prompt_version"),
              "taxonomy_version": header["taxonomy_version"],
              "exact_agreement": tagrun_rate(s["agree"], s["scored"]),
              "runner_up": ru, "issue_column": table}
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"analysis-{header['run_id']}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")

    def pct(r):
        lo, hi = r["wilson95"]
        return (f"{r['k']} of {r['n']} ({100 * r['k'] / max(r['n'], 1):.1f}%, 95% interval "
                f"{100 * lo:.1f} to {100 * hi:.1f}%)")

    out(f"Analysis of {header['run_id']} ({header['set']}, run at {header.get('code_commit')})")
    out(f"Exact agreement: {pct(result['exact_agreement'])}")
    out(f"Misses: {ru['misses']}; with a 'vs' note: {ru['with_runner_up_note']}; "
        f"tag equals the noted runner-up: {ru['match_runner_up']} of {ru['misses']}")
    out("  By product: " + "; ".join(f"{p} {g['match_runner_up']} of {g['misses']}"
                                     for p, g in ru["by_product"].items()))
    out("  By labelled code:")
    for code, g in ru["by_label"].items():
        out(f"    {code:<20} {g['match_runner_up']} of {g['misses']} misses")
    out(f"Issue column: {table['groups']} product-issue groups; "
        f"{table['groups_with_2plus_causes']} hold 2 or more root causes "
        f"({table['items_in_those_groups']} of {table['scored']} items)")
    b = table["issue_only_bound"]
    out(f"  Issue-only bound (majority label per group, fitted on these items): "
        f"{b['k']} of {b['n']} ({100 * b['k'] / max(b['n'], 1):.1f}%)")
    out(f"  Tagger exact agreement on the same items: {pct(table['tagger_exact'])}")
    out("  Largest groups (n, causes, majority, tagger agree):")
    for r in table["rows"][:10]:
        short = "CC" if r["product"] == "Credit card" else "CS"
        out(f"    {short} {r['issue'][:58]:<58} {r['n']:>3} {r['causes']:>2} "
            f"{r['majority']:>3} {r['tagger_agree']:>3}")
    out(f"Written: {path}")
    return path, result


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        analyse(argv[1])
    except (AnalysisError, taxonomy.TaxonomyError, OSError, sqlite3.Error) as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
