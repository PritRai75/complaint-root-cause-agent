"""Run the tagger on hand-labelled items and compare with Prit's labels.

Usage:
  python -m eval.tagrun dev [--limit N]      the 90 dev items (reading 60 + pilot 30)
  python -m eval.tagrun dev --fresh          the same, ignoring cached answers
  python -m eval.tagrun test --final          the 200 test items, once, at the end

Dev may be run as often as needed to tune the prompt. --fresh calls the model
again for every item, to measure how much tags move between identical runs at
temperature 0 (Project 1 saw real variation there). Test gives the published
numbers, so it needs --final. Every run writes eval/runs/<run_id>.jsonl: a
header line (commit, model, prompt version and hash, taxonomy version, set,
settings), one line per item (ref, IDs, label, tag, status, tokens, time; no
narrative text) and a summary line. Items marked uncertain are left out of
scoring (decision C, session 3); none are, so far.

The printout gives exact agreement as k of n with a Wilson 95% interval, by
product, by labelled code, the most frequent label -> tag confusions, and
tokens with a cost estimate. Calls go to Bedrock (Sonnet 4.6, ap-south-1)
with your AWS login; results are cached, so a re-run of the same prompt costs
nothing.
"""
import json
import math
import os
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone

from agent import semantic as sem
from agent import tagger as tg
from eval import taxonomy
from ingest.build_db import git_commit

LABELS_DIR = "eval/labels"
RUNS_DIR = "eval/runs"
SETS = {"dev": ("reading", "pilot"), "test": ("test",)}


class RunError(Exception):
    pass


def load_items(kind, labels_dir=LABELS_DIR):
    items = []
    for name in SETS[kind]:
        with open(os.path.join(labels_dir, f"{name}.toml"), "rb") as f:
            labels = tomllib.load(f)
        for it in labels["item"]:
            items.append({"ref": it["ref"], "complaint_id": it["complaint_id"],
                          "product": it["product"], "label": it["code"],
                          "uncertain": bool(it.get("uncertain", False)), "set": name,
                          "label_taxonomy": labels["taxonomy_version"]})
    return items


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score(rows):
    scored = [r for r in rows if not r["uncertain"]]
    tagged = [r for r in scored if r["status"] == "tagged"]
    right = sum(r["tag"] == r["label"] for r in tagged)
    lo, hi = wilson(right, len(scored))
    by_product = {}
    for p in sorted({r["product"] for r in scored}):
        sub = [r for r in scored if r["product"] == p]
        by_product[p] = [sum(r["tag"] == r["label"] for r in sub), len(sub)]
    by_label = {}
    for code in sorted({r["label"] for r in scored}):
        sub = [r for r in scored if r["label"] == code]
        by_label[code] = [sum(r["tag"] == r["label"] for r in sub), len(sub)]
    confusions = Counter((r["label"], r["tag"] or r["status"]) for r in scored
                         if r["tag"] != r["label"])
    return {"scored": len(scored), "agree": right, "wilson95": [round(lo, 3), round(hi, 3)],
            "not_tagged": len(scored) - len(tagged), "by_product": by_product,
            "by_label": by_label,
            "confusions": [[a, b, n] for (a, b), n in confusions.most_common()]}


def run(kind, final=False, limit=None, fresh=False, labels_dir=LABELS_DIR, runs_dir=RUNS_DIR,
        agent_path=sem.AGENT_DB_PATH, layer_path=sem.LAYER_PATH, taxonomy_path=None,
        client=None, cache_path=tg.CACHE_PATH, out=print):
    if kind not in SETS:
        raise RunError(f"unknown set {kind!r}: use dev or test")
    if kind == "test" and not final:
        raise RunError("the test split gives the published numbers: run it once, "
                       "at the end, with --final")
    if kind == "test" and limit:
        raise RunError("--limit is for dev only")
    tax = taxonomy.load(taxonomy_path)
    items = load_items(kind, labels_dir)
    stale = sorted({it["label_taxonomy"] for it in items} - {tax["version"]})
    if stale:
        raise RunError(f"labels were made under taxonomy {stale}, the taxonomy is now "
                       f"{tax['version']}")
    if limit:
        items = items[:limit]

    started = datetime.now(timezone.utc)
    run_id = f"{kind}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    header = {"type": "header", "run_id": run_id, "set": kind, "final": final,
              "limit": limit, "fresh": fresh, "items": len(items), "code_commit": git_commit(),
              "model": tg.MODEL_ID, "region": tg.REGION, "prompt_version": tg.PROMPT_VERSION,
              "prompt_sha256": tg.prompt_sha256(tax), "taxonomy_version": tax["version"],
              "temperature": tg.TEMPERATURE, "max_tokens": tg.MAX_TOKENS,
              "text_cap": tg.TEXT_CAP, "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ")}

    text_conn = sem.connect_text(agent_path, layer_path)
    tagger = tg.Tagger(tax, text_conn, client=client, cache_path=cache_path,
                       read_cache=not fresh)
    t0 = time.monotonic()
    try:
        results = tagger.tag([(it["complaint_id"], it["product"]) for it in items])
    finally:
        tagger.close()
        text_conn.close()
    elapsed = time.monotonic() - t0

    rows = []
    for it, r in zip(items, results):
        rows.append({"type": "item", "ref": it["ref"], "set": it["set"],
                     "complaint_id": it["complaint_id"], "product": it["product"],
                     "label": it["label"], "uncertain": it["uncertain"], "tag": r["code"],
                     "status": r["status"], "attempts": r["attempts"], "cached": r["cached"],
                     "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                     "seconds": r["seconds"], "cut": r["cut"], "error": r["error"]})
    s = score(rows)
    t_in = sum(r["input_tokens"] for r in rows if not r["cached"])
    t_out = sum(r["output_tokens"] for r in rows if not r["cached"])
    summary = {"type": "summary", **s,
               "status": dict(Counter(r["status"] for r in rows)),
               "cached": sum(r["cached"] for r in rows), "cut": sum(r["cut"] for r in rows),
               "new_input_tokens": t_in, "new_output_tokens": t_out,
               "est_cost_usd": round((t_in * tg.PRICE_IN + t_out * tg.PRICE_OUT) / 1e6, 2),
               "elapsed_seconds": round(elapsed, 1)}

    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_id}.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in [header] + rows + [summary]:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    lo, hi = s["wilson95"]
    out(f"Run {run_id}: {kind}, {len(items)} items, model {tg.MODEL_ID}, prompt "
        f"{tg.PROMPT_VERSION} ({header['prompt_sha256'][:12]}), commit {header['code_commit']}")
    out(f"Agreement with labels: {s['agree']} of {s['scored']} "
        f"({100 * s['agree'] / max(s['scored'], 1):.1f}%, 95% interval "
        f"{100 * lo:.1f} to {100 * hi:.1f}%). Not tagged: {s['not_tagged']}.")
    out("By product: " + "; ".join(f"{p} {k} of {n}" for p, (k, n) in s["by_product"].items()))
    out("By labelled code:")
    for code, (k, n) in s["by_label"].items():
        out(f"  {code:<20} {k} of {n}")
    if s["confusions"]:
        out("Label -> tag, where they differ:")
        for a, b, n in s["confusions"][:12]:
            out(f"  {a} -> {b}: {n}")
    out(f"Status {summary['status']}; cached {summary['cached']}; cut {summary['cut']}. "
        f"New tokens {t_in:,} in, {t_out:,} out, about ${summary['est_cost_usd']:.2f} "
        f"(list price, not checked against AWS). {summary['elapsed_seconds']} s.")
    out(f"Log: {path}")
    return path, summary


def main(argv):
    args = argv[1:]
    if not args or args[0] not in SETS:
        print(__doc__)
        return 2
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        try:
            limit = int(args[i + 1])
        except (IndexError, ValueError):
            print("ERROR --limit needs a number, e.g. --limit 3")
            return 2
    try:
        run(args[0], final="--final" in args, limit=limit, fresh="--fresh" in args)
    except (RunError, taxonomy.TaxonomyError, sem.LayerError) as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
