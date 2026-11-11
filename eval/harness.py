"""The agent eval harness: Prit's tasks, run through agent.graph.answer, graded by code.

Decisions (session 8, Prit):
  31  The 15 tasks (12 report + 3 clarify) are all held out: split = "test", run
      once with --final after the planner prompt is frozen. Tuning uses separate
      dev tasks (split = "dev") that are not counted.
  32  A report task's key figures must be written in the findings, rounded the
      way the report's number check allows. Chart tables alone don't count.
  33  Direction of change is checked with word lists (UP, DOWN below); Prit
      reviews every failure.
  34  Root-cause tasks: the narratives behind the report's tag set must be
      exactly the reference population's seeded draw.

Checks (all by code). A task passes when every check that applies passes.
  outcome     report: status "sent". clarify: status "needs_clarification".
  email       report: exactly one dry-run send, to the fixed recipient, with the
              report as the attachment. clarify: no send.
  figures     every graded cell of the reference result appears in the findings.
              With alternatives, one alternative found in full is enough.
  direction   the finding that gives the direction_of figure says it moved the
              expected way (see direction_ok).
  population  the report's tag set drew exactly the reference draw.
  causes      the findings give the shares of that tag set's top N root causes.
              Causes tied at the cut have the same share, so ties need no rule.
The clarifying question's wording is not graded: it is logged for Prit to read.

What a figure match means. Numbers: a number written in a finding matches a
reference value when report.supported() would accept it (26.33 as 26.3%, 26%
or 26.33; 1,239 as 1,200; a fraction as a percentage; sign ignored), except
that a year matches only itself. Text: the cell's text, case and spacing
ignored, as whole words, an "s" allowed at the end (credit card / credit
cards). A month written YYYY-MM also matches its name ("March" or "March 2026").
Known weak spots: a small number (under 10) can match an unrelated number in a
finding, and a figure is found wherever it is written, even in a sentence about
something else. `check` warns about small numbers.

Run log: eval/runs/agent-<split>-<time>.jsonl, a header (commit, model, plan
version and prompt SHA-256, layer version, tagger prompt, tasks version and
SHA-256, grader version), one line per task (references, the run's outputs,
checks, tokens, cost, time; no narrative text) and a summary. Reports are saved
to scratch/eval/<run id>/ for reading. `regrade` re-scores a stored run with
the current grader and no model calls.

Usage (on the laptop; dev and test call Bedrock):
  python -m eval.harness check                   validate eval/tasks.toml and print each reference
  python -m eval.harness dev [--only D01,D02]    run the dev tasks
  python -m eval.harness test --final            the 15 test tasks, once
  python -m eval.harness regrade RUN_ID          re-score a stored run, no model calls
  python -m eval.harness show RUN_ID TASK_ID     one task of a run in full
"""
import calendar
import glob
import hashlib
import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime, timezone

from agent import graph as gr
from agent import report as rp
from agent import semantic as sem
from agent import send as sd
from agent import tagger as tg
from agent import text as textmod
from agent import tools as tl
from eval.tagrun import wilson
from ingest.build_db import git_commit

TASKS_PATH = "eval/tasks.toml"
RUNS_DIR = "eval/runs"
REPORTS_DIR = "scratch/eval"
GRADER_VERSION = "g1"
FIXED_TO = "fixed-recipient@example.invalid"
ID = re.compile(r"^[A-Z]\d{2}$")
SPLITS = ("dev", "test")
KINDS = ("report", "clarify")
TEST_COUNTS = {"report": 12, "clarify": 3}           # decision 4
KEYS = {"id", "split", "kind", "question", "figures", "direction", "direction_of",
        "population", "causes", "note"}
MAX_FIGURE_ROWS = 12
MAX_CAUSES = 3
SMALL = 10                     # numeric figures below this get a warning in check

UP = re.compile(r"\b(rose|rise[sn]?|rising|increas(?:e|ed|es|ing)|gr[eo]w|grows|growing|grown|"
                r"higher|up(?!\s+to\b)|jump(?:ed|s)?|climb(?:ed|s)?|doubled|tripled)\b", re.I)
DOWN = re.compile(r"\b(fell|falls?|falling|fallen|decreas(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|"
                  r"drop(?:ped|s)?|dropping|lower|down|fewer|shr[au]nk|shrinks?|halved)\b", re.I)
CLAUSE = re.compile(r"[,;:()]|\s(?:while|whereas|but|although|though)\s", re.I)
MONTHS = {f"{m:02d}": calendar.month_name[m] for m in range(1, 13)}


class TaskError(Exception):
    pass


# ---------- tasks ----------

def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def alternatives(task):
    f = task.get("figures")
    if f is None:
        return []
    return [f] if isinstance(f, str) else list(f)


def load_tasks(path=TASKS_PATH):
    """The tasks file, checked. Returns {"version", "sha256", "tasks": [...]}."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    unknown = set(data) - {"version", "task"}
    if unknown:
        raise TaskError(f"unknown top-level keys: {', '.join(sorted(unknown))}")
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise TaskError("version must be set (e.g. version = \"tasks-1\")")
    tasks, seen, questions = [], set(), {}
    for i, t in enumerate(data.get("task", []), 1):
        where = f"task {i}" + (f" ({t.get('id')})" if isinstance(t.get("id"), str) else "")
        bad = set(t) - KEYS
        if bad:
            raise TaskError(f"{where}: unknown keys {', '.join(sorted(bad))}")
        tid = t.get("id")
        if not isinstance(tid, str) or not ID.match(tid):
            raise TaskError(f"{where}: id must be a capital letter and two digits, e.g. N01")
        if tid in seen:
            raise TaskError(f"{where}: id {tid} is used twice")
        seen.add(tid)
        if t.get("split") not in SPLITS:
            raise TaskError(f"{where}: split must be one of {', '.join(SPLITS)}")
        if t.get("kind") not in KINDS:
            raise TaskError(f"{where}: kind must be one of {', '.join(KINDS)}")
        q = t.get("question")
        if not isinstance(q, str) or not q.strip():
            raise TaskError(f"{where}: question is empty")
        if len(" ".join(q.split())) > gr.MAX_QUESTION:
            raise TaskError(f"{where}: question is over {gr.MAX_QUESTION} characters")
        key = " ".join(q.lower().split())
        if key in questions:
            raise TaskError(f"{where}: same question as {questions[key]}")
        questions[key] = tid
        f = t.get("figures")
        if f is not None:
            alts = [f] if isinstance(f, str) else f
            if not isinstance(alts, list) or not alts or not all(
                    isinstance(s, str) and s.strip() for s in alts):
                raise TaskError(f"{where}: figures must be one SQL string or a list of them")
        pop = t.get("population")
        if pop is not None and (not isinstance(pop, str) or not pop.strip()):
            raise TaskError(f"{where}: population must be one SQL string")
        if t["kind"] == "clarify":
            extra = [k for k in ("figures", "direction", "direction_of", "population", "causes")
                     if k in t]
            if extra:
                raise TaskError(f"{where}: a clarify task has no {', '.join(extra)}")
        else:
            if f is None and pop is None:
                raise TaskError(f"{where}: a report task needs figures, population or both")
        if ("direction" in t) != ("direction_of" in t):
            raise TaskError(f"{where}: direction and direction_of go together")
        if "direction" in t:
            if t["direction"] not in ("up", "down"):
                raise TaskError(f"{where}: direction must be \"up\" or \"down\"")
            if f is None:
                raise TaskError(f"{where}: direction needs figures (direction_of names a column)")
            if not isinstance(t["direction_of"], str) or t["direction_of"].startswith("_"):
                raise TaskError(f"{where}: direction_of must name a graded column of figures")
        if (pop is None) != ("causes" not in t):
            raise TaskError(f"{where}: population and causes go together")
        if "causes" in t:
            c = t["causes"]
            if isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= MAX_CAUSES:
                raise TaskError(f"{where}: causes must be a whole number from 1 to {MAX_CAUSES}")
        if "note" in t and not isinstance(t["note"], str):
            raise TaskError(f"{where}: note must be text")
        tasks.append(dict(t, question=" ".join(q.split())))
    return {"version": version.strip(), "sha256": file_sha256(path), "tasks": tasks}


def test_counts(tasks):
    got = {k: sum(t["split"] == "test" and t["kind"] == k for t in tasks) for k in KINDS}
    return got, got == TEST_COUNTS


# ---------- references ----------

def figure_cells(columns, rows):
    """Graded cells of a reference result: every cell of a column whose name
    doesn't start with "_"."""
    cells = []
    for r, row in enumerate(rows):
        for c, name in enumerate(columns):
            if name.startswith("_"):
                continue
            v = row[c]
            kind = "number" if isinstance(v, (int, float)) and not isinstance(v, bool) else "text"
            cells.append({"row": r, "column": name, "value": v, "kind": kind})
    return cells


def reference(task, conn, guard, text_ids):
    """The task's reference figures and population draw, from the agent's database."""
    out = {"figures": [], "population": None}
    for n, sql in enumerate(alternatives(task), 1):
        res = tl.run_sql(conn, guard, sql, max_rows=MAX_FIGURE_ROWS + 1)
        label = f"{task['id']} figures" + (f" alternative {n}" if len(alternatives(task)) > 1 else "")
        if not res["ok"]:
            raise TaskError(f"{label}: {res['error']}")
        if res["truncated"] or res["row_count"] > MAX_FIGURE_ROWS:
            raise TaskError(f"{label}: more than {MAX_FIGURE_ROWS} rows")
        if not res["rows"]:
            raise TaskError(f"{label}: no rows")
        cells = figure_cells(res["columns"], res["rows"])
        if not cells:
            raise TaskError(f"{label}: every column starts with _, so nothing is graded")
        if any(c["value"] is None or (c["kind"] == "text" and not str(c["value"]).strip())
               for c in cells):
            raise TaskError(f"{label}: a graded cell is empty")
        d = None
        if "direction_of" in task:
            if task["direction_of"] not in res["columns"]:
                raise TaskError(f"{label}: no column {task['direction_of']} for direction_of")
            d = res["rows"][0][res["columns"].index(task["direction_of"])]
            if isinstance(d, bool) or not isinstance(d, (int, float)) or d == 0:
                raise TaskError(f"{label}: {task['direction_of']} must be a non-zero number")
            if (d > 0) != (task["direction"] == "up"):
                raise TaskError(f"{label}: {task['direction_of']} is {d}, which disagrees with "
                                f"direction = \"{task['direction']}\"")
        out["figures"].append({"sql": sql, "columns": res["columns"], "rows": res["rows"],
                               "cells": cells, "direction_value": d})
    if task.get("population"):
        d = tg.draw(conn, guard, text_ids, task["population"])
        if not d["ok"]:
            raise TaskError(f"{task['id']} population: {d['error']}")
        if not d["drawn"]:
            raise TaskError(f"{task['id']} population: no narratives to tag")
        out["population"] = {"sql": task["population"], "selected": len(d["selected"]),
                             "with_narrative": len(d["with_text"]), "drawn": len(d["drawn"]),
                             "drawn_ids": d["drawn"],
                             "drawn_sha256": ids_sha256(d["drawn"])}
    return out


def ids_sha256(ids):
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def references(tasks, agent_path=sem.AGENT_DB_PATH, layer_path=sem.LAYER_PATH):
    conn, guard = sem.connect(agent_path=agent_path, layer_path=layer_path)
    text_conn = sem.connect_text(agent_path, layer_path)
    try:
        text_ids = {r[0] for r in text_conn.execute(
            f"SELECT complaint_id FROM {textmod.TEXT_TABLE}")}
        return {t["id"]: reference(t, conn, guard, text_ids) for t in tasks}
    finally:
        text_conn.close()
        conn.close()


# ---------- grading ----------

def number_found(value, written):
    """Does one written number (from report.numbers_in) match a reference value?"""
    _, w, decimals, is_pct = written
    if decimals == 0 and not is_pct and w.is_integer() and int(w) in rp.YEARS:
        return abs(float(value)) == w
    return rp.supported(w, decimals, is_pct, [float(value)])


def norm(text):
    return " ".join(str(text).lower().split())


def text_found(value, texts):
    v = norm(value)
    pats = [r"(?<!\w)" + re.escape(v) + r"s?(?!\w)"]
    m = re.fullmatch(r"(\d{4})-(\d{2})", v)
    if m and m.group(2) in MONTHS:
        pats.append(r"\b" + MONTHS[m.group(2)].lower() + r"\b")
    return any(re.search(p, norm(t)) for p in pats for t in texts)


def cell_found(cell, findings):
    if cell["kind"] == "number":
        return any(number_found(cell["value"], w) for f in findings for w in rp.numbers_in(f))
    return text_found(cell["value"], findings)


def fmt_cell(cell):
    return f"{cell['column']}={cell['value']}"


def clauses(text):
    return [c for c in CLAUSE.split(text) if c.strip()]


def direction_ok(expected, value, findings):
    """(ok, detail). Looks at each place the direction_of figure is written:
    the clause holding it (split at , ; : ( ) while whereas but although
    though) must use an expected-direction word and no opposite word; if the
    clause has neither, the whole finding is read the same way. Passes if any
    place passes."""
    want, other = (UP, DOWN) if expected == "up" else (DOWN, UP)
    places = 0
    for f in findings:
        for c in clauses(f):
            if not any(number_found(value, w) for w in rp.numbers_in(c)):
                continue
            places += 1
            text = c if (want.search(c) or other.search(c)) else f
            if want.search(text) and not other.search(text):
                return True, f"says {expected}: {text.strip()[:120]}"
    if not places:
        return False, f"the change figure {value} is not in the findings"
    return False, f"no finding giving {value} says it went {expected} without also saying the opposite"


def top_shares(info, n):
    """Shares of the top n root causes of a tag set (ties at the cut share a value)."""
    ranked = sorted(((c, k) for c, k in info["counts"].items()
                     if info["shares"].get(c) is not None and k > 0), key=lambda x: -x[1])
    if not ranked:
        return []
    cut = ranked[min(n, len(ranked)) - 1][1]
    return [{"code": c, "count": k, "share": info["shares"][c]} for c, k in ranked if k >= cut]


def grade(task, ref, out):
    """{check: {"ok": bool, "detail": str}} for one task's outputs."""
    checks = {}
    status = out.get("status")
    if task["kind"] == "clarify":
        checks["outcome"] = {"ok": status == "needs_clarification", "detail": f"status {status}"}
        checks["email"] = {"ok": not out.get("sent"), "detail": f"{len(out.get('sent') or [])} sends"}
        return checks
    checks["outcome"] = {"ok": status == "sent", "detail": f"status {status}"
                         + (f" ({out.get('stop_reason')})" if out.get("stop_reason") else "")}
    sends, rep = out.get("sent") or [], out.get("report")
    ok = (len(sends) == 1 and rep is not None and sends[0].get("to") == FIXED_TO
          and [a.get("sha256") for a in sends[0].get("attachments", [])] == [rep["sha256"]])
    checks["email"] = {"ok": ok, "detail": f"{len(sends)} sends"
                       + ("" if ok or not sends else ", wrong recipient or attachment")}
    findings = list(rep["findings"]) if rep else []
    if ref["figures"]:
        best = None
        for n, alt in enumerate(ref["figures"], 1):
            missing = [fmt_cell(c) for c in alt["cells"] if not cell_found(c, findings)]
            if best is None or len(missing) < len(best[1]):
                best = (n, missing)
            if not missing:
                break
        n, missing = best
        tag = f" (alternative {n})" if len(ref["figures"]) > 1 else ""
        checks["figures"] = {"ok": not missing,
                             "detail": ("all found" if not missing else "missing " + ", ".join(missing))
                             + tag}
        if "direction" in task:
            value = ref["figures"][n - 1]["direction_value"]
            d_ok, detail = direction_ok(task["direction"], value, findings)
            checks["direction"] = {"ok": d_ok, "detail": detail}
    if ref["population"]:
        handle = rep.get("tags") if rep else None
        info = (out.get("tag_sets") or {}).get(handle) if handle else None
        if info is None:
            checks["population"] = {"ok": False, "detail": "the report has no tag set"}
            checks["causes"] = {"ok": False, "detail": "no tag set"}
        else:
            same = sorted(info.get("drawn_ids") or []) == sorted(ref["population"]["drawn_ids"])
            checks["population"] = {"ok": same, "detail": (
                f"{handle} drew the reference {ref['population']['drawn']}" if same else
                f"{handle} drew {info.get('drawn')}, reference {ref['population']['drawn']}, "
                f"{len(set(info.get('drawn_ids') or []) & set(ref['population']['drawn_ids']))} shared")}
            if not same:
                checks["causes"] = {"ok": False, "detail": "not graded: different draw"}
            else:
                top = top_shares(info, task["causes"])
                missing = [f"{t['code']} {t['share']}%" for t in top
                           if not any(number_found(t["share"], w)
                                      for f in findings for w in rp.numbers_in(f))]
                checks["causes"] = {"ok": bool(top) and not missing,
                                    "detail": "all found" if top and not missing else
                                    ("no root causes tagged" if not top else
                                     "missing " + ", ".join(missing))}
    return checks


def passed(checks):
    return bool(checks) and all(c["ok"] for c in checks.values())


# ---------- running ----------

def outputs(result, run, sender):
    """What the harness keeps from one agent run (no narrative text)."""
    store = run.store if run else None
    tag_sets = {}
    if store:
        for h, info in store.info.items():
            tag_sets[h] = {k: info.get(k) for k in (
                "sql", "selected", "with_narrative", "drawn", "sampled", "seed", "status",
                "counts", "shares", "root_cause_base", "drawn_ids", "model_calls")}
    rep = result.get("report")
    return {
        "status": result.get("status"), "stop_reason": result.get("stop_reason"),
        "model_error": result.get("model_error"),
        "clarifying_question": result.get("clarifying_question"),
        "final_text": result.get("final_text"),
        "report": None if rep is None else {k: rep[k] for k in (
            "title", "findings", "charts", "tags", "filename", "sha256", "bytes")},
        "sent": list(sender.sent), "send_record": result.get("sent"),
        "queries": result.get("queries"), "tag_sets": tag_sets,
        "counts": result.get("counts"), "tokens": result.get("tokens"),
        "tool_trace": result.get("tool_trace"), "seconds": result.get("seconds"),
        "plan_version": result.get("plan_version"), "prompt_sha256": result.get("prompt_sha256"),
    }


def cost(out):
    t = out.get("tokens") or {}
    planner = (t.get("input", 0) * tg.PRICE_IN + t.get("output", 0) * tg.PRICE_OUT) / 1e6
    tag_in = sum((s.get("model_calls") or {}).get("input_tokens", 0)
                 for s in (out.get("tag_sets") or {}).values())
    tag_out = sum((s.get("model_calls") or {}).get("output_tokens", 0)
                  for s in (out.get("tag_sets") or {}).values())
    tagging = (tag_in * tg.PRICE_IN + tag_out * tg.PRICE_OUT) / 1e6
    return {"planner": round(planner, 4), "tagging": round(tagging, 4),
            "total": round(planner + tagging, 4)}


def summarise(lines):
    n = len(lines)
    k = sum(line["pass"] for line in lines)
    lo, hi = wilson(k, n)
    by_check = {}
    for line in lines:
        for name, c in line["checks"].items():
            got = by_check.setdefault(name, [0, 0])
            got[0] += c["ok"]
            got[1] += 1
    by_kind = {}
    for kind in KINDS:
        sub = [line for line in lines if line["kind"] == kind]
        if sub:
            by_kind[kind] = [sum(line["pass"] for line in sub), len(sub)]
    return {"type": "summary", "tasks": n, "passed": k, "wilson95": [round(lo, 3), round(hi, 3)],
            "by_kind": by_kind, "by_check": by_check,
            "cost": round(sum(line["cost"]["total"] for line in lines), 4),
            "seconds": round(sum((line["outputs"].get("seconds") or 0) for line in lines), 1)}


def aws_ready():
    """None if AWS credentials load, else a one-line reason."""
    try:
        import boto3
        if boto3.Session().get_credentials() is None:
            return "no AWS credentials found: run aws login first"
    except Exception as exc:                       # noqa: BLE001 (the type only)
        return f"AWS credentials could not be loaded ({type(exc).__name__}): run aws login first"
    return None


def run(split, final=False, only=None, tasks_path=TASKS_PATH, runs_dir=RUNS_DIR,
        reports_dir=REPORTS_DIR, planner_for=None, agent_kw=None, now=None, echo=print):
    """Runs one split's tasks and writes the run log. planner_for(task) gives a
    planner (tests); None means Bedrock. Returns the run log's path."""
    tf = load_tasks(tasks_path)
    tasks = [t for t in tf["tasks"] if t["split"] == split]
    if only:
        wanted = set(only)
        unknown = wanted - {t["id"] for t in tasks}
        if unknown:
            raise TaskError(f"not {split} tasks: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["id"] in wanted]
    if not tasks:
        raise TaskError(f"no {split} tasks in {tasks_path}")
    commit = git_commit()
    if split == "test":
        if not final:
            raise TaskError("the test tasks give the published numbers: add --final")
        if only:
            raise TaskError("the test run covers every test task: drop --only")
        got, ok = test_counts(tf["tasks"])
        if not ok:
            raise TaskError(f"test needs {TEST_COUNTS['report']} report and "
                            f"{TEST_COUNTS['clarify']} clarify tasks; found "
                            f"{got['report']} and {got['clarify']}")
        done = glob.glob(os.path.join(runs_dir, "agent-test-*.jsonl"))
        if done:
            raise TaskError(f"a test run already exists ({os.path.basename(done[0])}): "
                            "the test tasks run once")
        if commit.endswith("-dirty") or commit == "unknown":
            raise TaskError(f"commit is {commit}: commit every change first, so the "
                            "logged commit is the code that ran")
    elif final:
        raise TaskError("--final is only for the test split")
    agent_kw = dict(agent_kw or {})
    refs = references(tasks, agent_path=agent_kw.get("agent_path", sem.AGENT_DB_PATH),
                      layer_path=agent_kw.get("layer_path", sem.LAYER_PATH))
    if planner_for is None:
        reason = aws_ready()
        if reason:
            raise TaskError(reason)
    layer = sem.load(agent_kw.get("layer_path", sem.LAYER_PATH))
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"agent-{split}-{stamp}"
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, run_id + ".jsonl")
    header = {"type": "header", "run_id": run_id, "split": split, "final": final,
              "only": sorted(only) if only else None, "tasks": [t["id"] for t in tasks],
              "code_commit": commit, "model": gr.MODEL_ID, "region": gr.REGION,
              "plan_version": gr.PLAN_VERSION, "prompt_sha256": gr.prompt_sha256(layer),
              "layer_version": layer["version"], "tagger_prompt": tg.PROMPT_VERSION,
              "tasks_version": tf["version"], "tasks_sha256": tf["sha256"],
              "grader_version": GRADER_VERSION, "fixed_to": FIXED_TO,
              "started_at": stamp}
    lines = []
    with open(path, "w", encoding="utf-8") as log:
        log.write(json.dumps(header) + "\n")
        log.flush()
        for task in tasks:
            sender = sd.DryRunSender(to=FIXED_TO)
            planner = planner_for(task) if planner_for else None
            started = time.monotonic()
            try:
                result, agent_run = gr.answer(task["question"], sender, planner=planner,
                                              **agent_kw)
            except Exception as exc:                     # noqa: BLE001 (the type only)
                result, agent_run = {"status": "harness_error",
                                     "stop_reason": type(exc).__name__}, None
            try:
                out = outputs(result, agent_run, sender)
                if agent_run and agent_run.store.report:
                    folder = os.path.join(reports_dir, run_id)
                    os.makedirs(folder, exist_ok=True)
                    with open(os.path.join(folder, f"{task['id']}.docx"), "wb") as f:
                        f.write(agent_run.store.report["bytes"])
            finally:
                if agent_run:
                    agent_run.close()
            if out.get("seconds") is None:
                out["seconds"] = round(time.monotonic() - started, 1)
            checks = grade(task, refs[task["id"]], out)
            line = {"type": "task", "id": task["id"], "split": split, "kind": task["kind"],
                    "question": task["question"], "task_fields": task_fields(task),
                    "reference": refs[task["id"]],
                    "outputs": out, "checks": checks, "pass": passed(checks), "cost": cost(out)}
            lines.append(line)
            log.write(json.dumps(line) + "\n")
            log.flush()
            echo(task_line(line))
        summary = summarise(lines)
        log.write(json.dumps(summary) + "\n")
    echo(summary_text(summary))
    echo(f"Run log: {path}")
    return path


def task_line(line):
    failed = [f"{n}: {c['detail']}" for n, c in line["checks"].items() if not c["ok"]]
    out = line["outputs"]
    head = (f"{line['id']} {line['kind']:<7} {'PASS' if line['pass'] else 'FAIL'}  "
            f"${line['cost']['total']:.2f}  {out.get('seconds') or 0:.0f} s")
    return head + "".join(f"\n    {f}" for f in failed)


def summary_text(s):
    lo, hi = s["wilson95"]
    parts = [f"{s['passed']} of {s['tasks']} tasks passed (Wilson 95% {100 * lo:.1f} to "
             f"{100 * hi:.1f}%)"]
    parts += [f"  {kind}: {k} of {n}" for kind, (k, n) in s["by_kind"].items()]
    parts += [f"  check {name}: {k} of {n}" for name, (k, n) in s["by_check"].items()]
    parts.append(f"  cost about ${s['cost']:.2f}, {s['seconds']:.0f} s")
    return "\n".join(parts)


# ---------- stored runs ----------

def read_run(run_id, runs_dir=RUNS_DIR):
    path = run_id if run_id.endswith(".jsonl") else os.path.join(runs_dir, run_id + ".jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or rows[0].get("type") != "header":
        raise TaskError(f"{path} is not a harness run log")
    return path, rows


def regrade(run_id, runs_dir=RUNS_DIR, echo=print):
    """Re-scores a stored run with the current grader: stored references and
    outputs, no database and no model calls. Writes <run>-regraded-<grader>.jsonl."""
    path, rows = read_run(run_id, runs_dir)
    header = dict(rows[0], regraded_from=os.path.basename(path), grader_version=GRADER_VERSION)
    lines, changed = [], 0
    for row in rows[1:]:
        if row.get("type") != "task":
            continue
        checks = grade(task_for(row), row["reference"], row["outputs"])
        new = dict(row, checks=checks, **{"pass": passed(checks)})
        changed += new["pass"] != row["pass"]
        lines.append(new)
    out_path = path[:-len(".jsonl")] + f"-regraded-{GRADER_VERSION}.jsonl"
    summary = summarise(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.write(json.dumps(summary) + "\n")
    echo(summary_text(summary))
    echo(f"{changed} tasks changed pass/fail. Written: {out_path}")
    return out_path


def task_fields(task):
    return {k: task[k] for k in ("direction", "direction_of", "causes") if k in task}


def task_for(row):
    """The task fields grade() reads, rebuilt from a stored line."""
    t = {"id": row["id"], "kind": row["kind"]}
    t.update(row.get("task_fields") or {})
    return t


def show(run_id, task_id, runs_dir=RUNS_DIR, echo=print):
    _, rows = read_run(run_id, runs_dir)
    row = next((r for r in rows if r.get("type") == "task" and r["id"] == task_id), None)
    if row is None:
        raise TaskError(f"no task {task_id} in {run_id}")
    out = row["outputs"]
    echo(task_line(row))
    echo(f"\nQuestion: {row['question']}")
    if out.get("clarifying_question"):
        echo(f"Clarifying question: {out['clarifying_question']}")
    rep = out.get("report")
    if rep:
        echo(f"\nTitle: {rep['title']}")
        for f in rep["findings"]:
            echo(f"- {f}")
    for n, alt in enumerate(row["reference"]["figures"], 1):
        echo(f"\nReference {n}: {' '.join(alt['sql'].split())}")
        for c in alt["cells"]:
            echo(f"    {fmt_cell(c)}")
    pop = row["reference"].get("population")
    if pop:
        echo(f"\nReference population: {pop['drawn']} drawn of {pop['with_narrative']} with a "
             f"narrative ({pop['selected']} selected)")
    for name, c in row["checks"].items():
        echo(f"{name:<10} {'ok' if c['ok'] else 'FAIL'}  {c['detail']}")
    for h, sql in (out.get("queries") or {}).items():
        echo(f"\n{h}: {' '.join(sql.split())}")
    for h, s in (out.get("tag_sets") or {}).items():
        echo(f"\n{h}: {' '.join((s.get('sql') or '').split())}  (drew {s.get('drawn')})")


# ---------- check ----------

def check(tasks_path=TASKS_PATH, agent_path=sem.AGENT_DB_PATH, layer_path=sem.LAYER_PATH,
          echo=print):
    """Validates the tasks file and prints every reference, before any agent run."""
    tf = load_tasks(tasks_path)
    tasks = tf["tasks"]
    got, ok = test_counts(tasks)
    echo(f"{tasks_path}: {tf['version']}, {len(tasks)} tasks "
         f"(dev {sum(t['split'] == 'dev' for t in tasks)}, test {got['report']} report + "
         f"{got['clarify']} clarify{'' if ok else ' (the test run needs 12 + 3)'})")
    refs = references(tasks, agent_path=agent_path, layer_path=layer_path)
    warnings = 0
    for t in tasks:
        r = refs[t["id"]]
        echo(f"\n{t['id']} {t['split']} {t['kind']}: {t['question']}")
        for n, alt in enumerate(r["figures"], 1):
            label = "figures" + (f" alt {n}" if len(r["figures"]) > 1 else "")
            echo(f"  {label}: " + ", ".join(fmt_cell(c) for c in alt["cells"]))
            for c in alt["cells"]:
                if c["kind"] == "number" and abs(c["value"]) < SMALL:
                    warnings += 1
                    echo(f"  warning: {fmt_cell(c)} is a small number and can match an "
                         "unrelated number in a finding")
            if alt["direction_value"] is not None:
                echo(f"  direction: {t['direction']} ({t['direction_of']} = {alt['direction_value']})")
        if r["population"]:
            p = r["population"]
            echo(f"  population: {p['selected']:,} selected, {p['with_narrative']:,} with a "
                 f"narrative, {p['drawn']} drawn; top {t['causes']} causes graded")
    echo(f"\n{len(tasks)} tasks checked, {warnings} warnings.")
    return refs


def main(argv):
    args = argv[1:]
    try:
        if args[:1] == ["check"] and len(args) == 1:
            check()
        elif args[:1] in (["dev"], ["test"]):
            final = "--final" in args
            rest = [a for a in args[1:] if a != "--final"]
            only = None
            if rest[:1] == ["--only"] and len(rest) == 2:
                only = [s.strip() for s in rest[1].split(",") if s.strip()]
            elif rest:
                print(__doc__)
                return 2
            run(args[0], final=final, only=only)
        elif args[:1] == ["regrade"] and len(args) == 2:
            regrade(args[1])
        elif args[:1] == ["show"] and len(args) == 3:
            show(args[1], args[2])
        else:
            print(__doc__)
            return 2
    except (TaskError, sem.LayerError, FileNotFoundError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
