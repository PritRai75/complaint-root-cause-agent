"""Per-class precision, recall and F1 of a tagger run, and its confusion matrix.

Usage:
  python -m eval.per_class eval/runs/<run_id>.jsonl

Reads a run log written by eval.tagrun and Prit's labels (no database, no
model calls), prints the table and writes eval/runs/per-class-<run_id>.json
and docs/img/confusion-<run_id>.png. For the test split it takes only the
--final log. A dev log is accepted for checking.

This re-scores a frozen log; no tag changes and nothing is run again. Exact
match stays the only scoring rule (decisions 20 and 54): a tag is right only
when it equals the label, and every figure here is that same comparison
split by code. Rules (decision 54, fixed 23 Nov 2026, after the test run's
totals were known):

- Precision of a code = right tags of that code / items tagged with it.
  Recall = right tags / items labelled with it. F1 = their harmonic mean,
  0 when both are 0.
- An item the tagger didn't tag (any status other than "tagged") counts as a
  miss for its label's recall and for no code's precision.
- Macro-F1 averages F1 over all 10 codes of the taxonomy (non-causes
  included). Micro precision, recall and F1 pool all items; micro recall is
  exact agreement, and micro-F1 equals it when every item was tagged.
- Wilson 95% intervals on precision and on recall, each only when its own
  denominator is 10 or more; smaller ones print as k of n marked "low
  support". No interval on F1 (it isn't a proportion).
- Count bias of a code = tagged / labelled = recall / precision: above 1 the
  code is over-counted, below 1 under-counted.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval import taganalysis as ta  # noqa: E402
from eval import tagrun  # noqa: E402
from eval import taxonomy  # noqa: E402

IMG_DIR = "docs/img"
MIN_SUPPORT = 10
UNTAGGED = "(not tagged)"


class PerClassError(Exception):
    pass


def rate(k, n):
    """k of n with a Wilson 95% interval when n >= MIN_SUPPORT."""
    out = {"k": k, "n": n, "value": round(k / n, 4) if n else None,
           "low_support": n < MIN_SUPPORT}
    if n >= MIN_SUPPORT:
        lo, hi = tagrun.wilson(k, n)
        out["wilson95"] = [round(lo, 3), round(hi, 3)]
    return out


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def confusion(items, codes):
    """{label: {tag or UNTAGGED: count}} over scored items, every code a row."""
    cols = list(codes) + [UNTAGGED]
    m = {a: {b: 0 for b in cols} for a in codes}
    for it in items:
        if it["label"] not in m:
            raise PerClassError(f"{it['ref']}: label {it['label']!r} is not a taxonomy code")
        tag = it["tag"] if it["status"] == "tagged" else UNTAGGED
        if tag not in m[it["label"]]:
            raise PerClassError(f"{it['ref']}: tag {tag!r} is not a taxonomy code")
        m[it["label"]][tag] += 1
    return m


def per_class(items, codes):
    scored = [it for it in items if not it["uncertain"]]
    if not scored:
        raise PerClassError("no scored items")
    m = confusion(scored, codes)
    rows = []
    for c in codes:
        labelled = sum(m[c].values())
        tagged = sum(m[a][c] for a in codes)
        right = m[c][c]
        p = right / tagged if tagged else 0.0
        r = right / labelled if labelled else 0.0
        rows.append({"code": c, "labelled": labelled, "tagged": tagged, "right": right,
                     "precision": rate(right, tagged), "recall": rate(right, labelled),
                     "f1": round(f1(p, r), 4),
                     "count_bias": round(tagged / labelled, 3) if labelled else None})
    n = len(scored)
    n_tagged = sum(it["status"] == "tagged" for it in scored)
    right = sum(m[c][c] for c in codes)
    micro_p = right / n_tagged if n_tagged else 0.0
    micro_r = right / n
    return {"scored": n, "tagged": n_tagged, "right": right, "rows": rows,
            "macro_f1": round(sum(r["f1"] for r in rows) / len(rows), 4),
            "macro_f1_codes": len(rows),
            "micro": {"precision": rate(right, n_tagged), "recall": rate(right, n),
                      "f1": round(f1(micro_p, micro_r), 4)},
            "matrix": m}


def spotlight(result, code):
    """Where a code's false positives come from and where its misses go."""
    m = result["matrix"]
    row = next(r for r in result["rows"] if r["code"] == code)
    fp = {a: m[a][code] for a in m if a != code and m[a][code]}
    fn = {b: k for b, k in m[code].items() if b != code and k}
    return {"code": code, "precision": row["precision"], "recall": row["recall"],
            "f1": row["f1"],
            "false_positives_by_label": dict(sorted(fp.items(), key=lambda kv: (-kv[1], kv[0]))),
            "misses_by_tag": dict(sorted(fn.items(), key=lambda kv: (-kv[1], kv[0])))}


def plot_matrix(result, codes, path, title):
    m = result["matrix"]
    cols = list(codes) + ([UNTAGGED] if any(m[a][UNTAGGED] for a in codes) else [])
    data = [[m[a][b] for b in cols] for a in codes]
    # Colour by share of the row (so the diagonal shade is recall); counts as text.
    share = [[v / (sum(r) or 1) for v in r] for r in data]
    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=120)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if share[i][j] > 0.55 else "#1a1a1a",
                        fontweight="bold" if i == j else "normal")
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Share of the hand label's row (diagonal = recall)", fontsize=9)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(codes)), codes, fontsize=9)
    ax.set_xlabel("Tagger's code")
    ax.set_ylabel("Hand label")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, metadata={"Software": None})   # no version stamp: same bytes each run
    plt.close(fig)


def analyse(log_path, labels_dir=tagrun.LABELS_DIR, taxonomy_path=None,
            runs_dir=tagrun.RUNS_DIR, img_dir=IMG_DIR, out=print):
    header, items, summary = ta.read_log(log_path)
    tax = taxonomy.load(taxonomy_path)
    if header["taxonomy_version"] != tax["version"]:
        raise PerClassError(f"the run used taxonomy {header['taxonomy_version']}, the "
                            f"taxonomy is now {tax['version']}")
    codes = [c["code"] for c in tax["cause"]]
    notes = ta.load_notes(header["set"], labels_dir)
    ta.check_labels(items, notes)
    s = tagrun.score(items)
    if s["agree"] != summary["agree"] or s["scored"] != summary["scored"]:
        raise PerClassError("re-scoring the log doesn't give its own summary")

    res = per_class(items, codes)
    if res["right"] != s["agree"]:
        raise PerClassError("the matrix diagonal doesn't add up to exact agreement")
    run_id = header["run_id"]
    img = os.path.join(img_dir, f"confusion-{run_id}.png")
    plot_matrix(res, codes, img,
                f"{run_id}: hand label vs tag, {res['right']} of {res['scored']} on the diagonal")

    result = {"type": "per_class", "run_id": run_id, "set": header["set"],
              "final": header.get("final", False), "log_sha256": ta.sha256_file(log_path),
              "run_commit": header.get("code_commit"), "analysis_commit": tagrun.git_commit(),
              "prompt_version": header.get("prompt_version"),
              "taxonomy_version": header["taxonomy_version"], "min_support": MIN_SUPPORT,
              "exact_agreement": ta.tagrun_rate(s["agree"], s["scored"]),
              **{k: res[k] for k in ("scored", "tagged", "right", "macro_f1",
                                     "macro_f1_codes", "micro", "rows", "matrix")},
              "spotlight": spotlight(res, "fraud_claim"), "image": img.replace(os.sep, "/")}
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"per-class-{run_id}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")

    def show(r):
        txt = f"{r['k']:>3}/{r['n']:<3}"
        if r["n"] == 0:
            return f"{txt} {'-':>6} {'':<13}"
        pct = f"{100 * r['value']:5.1f}%"
        if r["low_support"]:
            return f"{txt} {pct} {'low support':<13}"
        lo, hi = r["wilson95"]
        return f"{txt} {pct} {f'{100 * lo:.1f}-{100 * hi:.1f}':<13}"

    out(f"Per-class scores of {run_id} ({header['set']}, run at {header.get('code_commit')})")
    out(f"Exact agreement {s['agree']} of {s['scored']}; tagged {res['tagged']}")
    out(f"{'code':<20} {'precision':<27} {'recall':<27} {'F1':>6} {'bias':>6}")
    for r in res["rows"]:
        bias = f"{r['count_bias']:.2f}" if r["count_bias"] is not None else "-"
        out(f"{r['code']:<20} {show(r['precision'])} {show(r['recall'])} "
            f"{100 * r['f1']:5.1f}% {bias:>6}")
    mi = res["micro"]
    out(f"Macro-F1 over {res['macro_f1_codes']} codes: {100 * res['macro_f1']:.1f}%")
    out(f"Micro precision {show(mi['precision']).strip()}; recall "
        f"{show(mi['recall']).strip()}; F1 {100 * mi['f1']:.1f}%")
    sp = result["spotlight"]
    out(f"fraud_claim: precision {show(sp['precision']).strip()}; recall "
        f"{show(sp['recall']).strip()}")
    out("  tagged fraud_claim but labelled: " + (", ".join(
        f"{c} {k}" for c, k in sp["false_positives_by_label"].items()) or "none"))
    out("  labelled fraud_claim but tagged: " + (", ".join(
        f"{c} {k}" for c, k in sp["misses_by_tag"].items()) or "none"))
    out(f"Written: {path}")
    out(f"Written: {img}")
    return path, result


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        analyse(argv[1])
    except (PerClassError, ta.AnalysisError, taxonomy.TaxonomyError, OSError) as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
