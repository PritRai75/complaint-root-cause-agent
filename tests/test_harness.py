"""The eval harness: task file checks, the grader, and whole runs on the fake
database with a scripted planner (no Bedrock)."""
import json
import os
import textwrap

import pytest

from agent import report as rp
from eval import harness as hs
from tests.test_graph import FINDINGS, ScriptedPlanner, call, happy_script, paths, turn  # noqa: F401


def write(tmp_path, body, version='version = "tasks-t"\n'):
    p = tmp_path / "tasks.toml"
    p.write_text(version + textwrap.dedent(body), encoding="utf-8")
    return str(p)


def task(tid="D01", split="dev", kind="report", question=None, **fields):
    lines = ["[[task]]", f'id = "{tid}"', f'split = "{split}"', f'kind = "{kind}"',
             f'question = """{question or "Question " + tid}"""']
    for k, v in fields.items():
        lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n\n"


CARD_COUNT = "SELECT COUNT(*) AS n FROM complaints_v WHERE product = 'Credit card'"


# ---------- the tasks file ----------

def test_the_committed_tasks_file_loads():
    tf = hs.load_tasks()
    assert tf["version"] and len(tf["sha256"]) == 64
    got, _ = hs.test_counts(tf["tasks"])
    assert got["report"] <= 12 and got["clarify"] <= 3


@pytest.mark.parametrize("body, message", [
    (task(tid="d1", figures=CARD_COUNT), "capital letter"),
    (task(figures=CARD_COUNT) + task(question="Other", figures=CARD_COUNT), "used twice"),
    (task() , "needs figures, population"),
    (task(kind="clarify", figures=CARD_COUNT), "clarify task has no figures"),
    (task(figures=CARD_COUNT, direction="up"), "go together"),
    (task(figures=CARD_COUNT, direction="sideways", direction_of="n"), '"up" or "down"'),
    (task(population="SELECT complaint_id FROM narratives_v"), "population and causes"),
    (task(population="SELECT complaint_id FROM narratives_v", causes=4), "1 to 3"),
    (task(figures=CARD_COUNT, colour="blue"), "unknown keys colour"),
    (task(split="train", figures=CARD_COUNT), "split must be"),
    (task(figures=[]), "one SQL string or a list"),
    (task(figures=CARD_COUNT) + task(tid="D02", question="question d01", figures=CARD_COUNT),
     "same question as D01"),
    (task(figures=CARD_COUNT, question="x" * 1001), "over 1000 characters"),
])
def test_bad_tasks_are_refused(tmp_path, body, message):
    with pytest.raises(hs.TaskError, match=message):
        hs.load_tasks(write(tmp_path, body))


def test_a_version_is_required(tmp_path):
    with pytest.raises(hs.TaskError, match="version"):
        hs.load_tasks(write(tmp_path, task(figures=CARD_COUNT), version=""))


# ---------- matching ----------

def nums(text):
    return rp.numbers_in(text)


def test_numbers_match_with_the_number_checks_rounding():
    assert hs.number_found(26.33, nums("26.3%")[0])
    assert hs.number_found(26.33, nums("26%")[0])
    assert hs.number_found(0.2633, nums("26.3%")[0])
    assert hs.number_found(1239, nums("1,200")[0])
    assert hs.number_found(-12.5, nums("12.5%")[0])            # sign ignored
    assert not hs.number_found(26.33, nums("26.4%")[0])
    assert not hs.number_found(1239, nums("1,000")[0])


def test_a_year_matches_only_itself():
    # report.supported() lets any year through; a reference figure must not.
    assert not hs.number_found(512, nums("in 2026")[0])
    assert hs.number_found(2026, nums("in 2026")[0])


def test_text_matches_as_whole_words():
    assert hs.text_found("Credit card", ["Credit cards drew the most complaints."])
    assert hs.text_found("Credit card", ["CREDIT  CARD complaints"])
    assert not hs.text_found("card", ["Cardholders complained."])
    assert hs.text_found("2026-03", ["Complaints peaked in March."])
    assert hs.text_found("2026-03", ["Complaints peaked in 2026-03."])
    assert not hs.text_found("2026-03", ["Complaints peaked in May."])


@pytest.mark.parametrize("expected, value, findings, ok", [
    ("up", 12.5, ["Card complaints rose 12.5% from January to June."], True),
    ("up", 12.5, ["Card complaints rose 12.5%, while checking complaints fell 3%."], True),
    ("down", 3, ["Card complaints rose 12.5%, while checking complaints fell 3%."], True),
    ("up", 3, ["Card complaints rose 12.5%, while checking complaints fell 3%."], False),
    ("up", 12.5, ["Card complaints changed by 12.5% and rose then fell."], False),
    ("up", 12.5, ["Card complaints went up, by 12.5%."], True),     # clause has no word: whole finding
    ("down", 12.5, ["Card complaints fell 12.5% to 900."], True),
    ("up", 12.5, ["A sample of up to 200 narratives; complaints changed 12.5%."], False),
    ("up", 12.5, ["Card complaints rose 10%."], False),             # the figure isn't there
])
def test_direction(expected, value, findings, ok):
    assert hs.direction_ok(expected, value, findings)[0] is ok


def test_top_shares_keep_ties_at_the_cut():
    info = {"counts": {"fraud_claim": 5, "fee_charged": 3, "servicing_error": 3,
                       "third_party": 9, "other": 0},
            "shares": {"fraud_claim": 45.5, "fee_charged": 27.3, "servicing_error": 27.3,
                       "third_party": None, "other": None}}
    assert [t["code"] for t in hs.top_shares(info, 1)] == ["fraud_claim"]
    assert {t["code"] for t in hs.top_shares(info, 2)} == {"fraud_claim", "fee_charged",
                                                          "servicing_error"}
    assert len(hs.top_shares(info, 3)) == 3


# ---------- grading stored outputs ----------

def report_out(findings, sends=1, to=hs.FIXED_TO, sha="abc", tags=None, tag_sets=None):
    return {"status": "sent", "report": {"title": "T", "findings": findings, "tags": tags,
                                         "sha256": sha},
            "sent": [{"to": to, "attachments": [{"sha256": sha}]}] * sends,
            "tag_sets": tag_sets or {}}


def ref_with(cells_rows, columns, direction_value=None):
    return {"figures": [{"sql": "x", "columns": columns, "rows": cells_rows,
                         "cells": hs.figure_cells(columns, cells_rows),
                         "direction_value": direction_value}], "population": None}


def test_grade_a_report_task():
    t = {"id": "D01", "kind": "report"}
    ref = ref_with([["Credit card", 1239, 26.33, "skip me"]], ["product", "n", "rate", "_note"])
    ok = hs.grade(t, ref, report_out(["Credit card complaints: 1,239, of which 26.3% got relief."]))
    assert hs.passed(ok) and set(ok) == {"outcome", "email", "figures"}
    miss = hs.grade(t, ref, report_out(["Credit card complaints: 1,239."]))
    assert not miss["figures"]["ok"] and "rate=26.33" in miss["figures"]["detail"]


def test_figures_in_the_title_only_dont_count():
    t = {"id": "D01", "kind": "report"}
    out = report_out(["Complaints were counted."])
    out["report"]["title"] = "1,239 card complaints"
    assert not hs.grade(t, ref_with([[1239]], ["n"]), out)["figures"]["ok"]


def test_any_one_alternative_is_enough():
    t = {"id": "D01", "kind": "report"}
    ref = ref_with([[100]], ["n"])
    ref["figures"].append(ref_with([[250]], ["n"])["figures"][0])
    g = hs.grade(t, ref, report_out(["There were 250 complaints."]))
    assert g["figures"]["ok"] and "alternative 2" in g["figures"]["detail"]


@pytest.mark.parametrize("out, why", [
    (report_out(["n is 5"], sends=0), "0 sends"),
    (report_out(["n is 5"], sends=2), "2 sends"),
    (report_out(["n is 5"], to="someone@else.example"), "wrong recipient"),
])
def test_email_must_be_exactly_one_to_the_fixed_recipient(out, why):
    g = hs.grade({"id": "D01", "kind": "report"}, ref_with([[5]], ["n"]), out)
    assert not g["email"]["ok"] and why in g["email"]["detail"]


def test_attachment_must_be_the_report():
    out = report_out(["n is 5"])
    out["sent"] = [{"to": hs.FIXED_TO, "attachments": [{"sha256": "other"}]}]
    assert not hs.grade({"id": "D01", "kind": "report"}, ref_with([[5]], ["n"]), out)["email"]["ok"]


def test_grade_a_clarify_task():
    t = {"id": "A01", "kind": "clarify"}
    asked = {"status": "needs_clarification", "sent": [], "clarifying_question": "Which year?"}
    assert hs.passed(hs.grade(t, {"figures": [], "population": None}, asked))
    built = report_out(["n is 5"])
    g = hs.grade(t, {"figures": [], "population": None}, built)
    assert not g["outcome"]["ok"] and not g["email"]["ok"]


def test_grade_direction():
    t = {"id": "D01", "kind": "report", "direction": "down", "direction_of": "change"}
    ref = ref_with([[-12.5]], ["change"], direction_value=-12.5)
    assert hs.grade(t, ref, report_out(["Complaints fell 12.5%."]))["direction"]["ok"]
    assert not hs.grade(t, ref, report_out(["Complaints rose 12.5%."]))["direction"]["ok"]


def pop_ref(ids):
    return {"figures": [], "population": {"drawn": len(ids), "drawn_ids": ids}}


def tag_set(ids, counts, shares):
    return {"tags_1": {"drawn": len(ids), "drawn_ids": ids, "counts": counts, "shares": shares}}


def test_grade_population_and_causes():
    t = {"id": "D01", "kind": "report", "causes": 1}
    sets = tag_set([3, 1, 2], {"fraud_claim": 2, "fee_charged": 1},
                   {"fraud_claim": 66.7, "fee_charged": 33.3})
    good = hs.grade(t, pop_ref([1, 2, 3]),
                    report_out(["Fraud was 66.7% of tagged narratives."], tags="tags_1",
                               tag_sets=sets))
    assert good["population"]["ok"] and good["causes"]["ok"]
    t2 = dict(t, causes=2)
    assert not hs.grade(t2, pop_ref([1, 2, 3]), report_out(
        ["Fraud was 66.7% of tagged narratives."], tags="tags_1", tag_sets=sets))["causes"]["ok"]
    other = hs.grade(t, pop_ref([1, 2, 4]), report_out(
        ["Fraud was 66.7%."], tags="tags_1", tag_sets=sets))
    assert not other["population"]["ok"] and "2 shared" in other["population"]["detail"]
    assert not other["causes"]["ok"]
    none = hs.grade(t, pop_ref([1, 2, 3]), report_out(["Fraud was 66.7%."]))
    assert not none["population"]["ok"]


# ---------- whole runs on the fake database ----------

ROOT = "SELECT complaint_id FROM narratives_v"


def tasks_for_run(tmp_path, extra=""):
    body = (task("D01", figures="SELECT COUNT(*) AS n FROM complaints_v WHERE product = "
                                "'Credit card'", population=ROOT, causes=1)
            + task("D02", kind="clarify") + extra)
    return write(tmp_path, body)


def planner_for(task):
    if task["kind"] == "clarify":
        return ScriptedPlanner(script=[turn(call("ask_clarifying_question",
                                                 question="Which **products** do you mean?"))])
    return ScriptedPlanner(script=happy_script())


def run_dev(tmp_path, paths, tasks_path=None, **kw):
    agent_kw = {k: paths[k] for k in ("agent_path", "layer_path", "tag_client", "tag_cache",
                                      "tag_sleep")}
    printed = []
    path = hs.run("dev", tasks_path=tasks_path or tasks_for_run(tmp_path),
                  runs_dir=str(tmp_path / "runs"), reports_dir=str(tmp_path / "reports"),
                  planner_for=planner_for, agent_kw=agent_kw, echo=printed.append, **kw)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return path, rows, printed


def test_a_dev_run_grades_and_logs(tmp_path, paths):
    path, rows, printed = run_dev(tmp_path, paths)
    header, d01, d02, summary = rows
    assert header["type"] == "header" and header["plan_version"] == "plan-2"
    assert header["grader_version"] == hs.GRADER_VERSION and header["tasks"] == ["D01", "D02"]
    assert len(header["prompt_sha256"]) == 64 and header["tasks_version"] == "tasks-t"
    # D01: the happy script reports 'Each product has 5 complaints' and fraud at 42.9%
    assert d01["pass"], d01["checks"]
    assert set(d01["checks"]) == {"outcome", "email", "figures", "population", "causes"}
    assert d01["reference"]["population"]["drawn"] == 8
    ts = d01["outputs"]["tag_sets"]["tags_1"]
    assert ts["drawn_ids"] == d01["reference"]["population"]["drawn_ids"]
    assert ts["model_calls"]["calls"] == 8                   # the first run tags uncached
    assert d01["cost"]["tagging"] > 0 and d01["outputs"]["report"]["findings"] == FINDINGS
    assert os.path.exists(tmp_path / "reports" / header["run_id"] / "D01.docx")
    # D02 asked a question and sent nothing
    assert d02["pass"] and d02["outputs"]["clarifying_question"] == "Which products do you mean?"
    assert summary["passed"] == 2 and summary["by_check"]["outcome"] == [2, 2]
    assert any("2 of 2 tasks passed" in p for p in printed)
    text = open(path, encoding="utf-8").read()
    assert "Someone used my card" not in text                # no narrative text in the log


def test_a_repeat_run_tags_from_the_cache(tmp_path, paths):
    run_dev(tmp_path, paths)
    _, rows, _ = run_dev(tmp_path, paths)
    assert rows[1]["outputs"]["tag_sets"]["tags_1"]["model_calls"]["calls"] == 0


def test_a_wrong_reference_fails_the_task(tmp_path, paths):
    tp = write(tmp_path, task("D01", figures="SELECT 7 AS n, 'Checking or savings account' "
                                             "AS product"))
    _, rows, printed = run_dev(tmp_path, paths, tasks_path=tp)
    assert not rows[1]["pass"] and "missing n=7" in rows[1]["checks"]["figures"]["detail"]
    assert any("FAIL" in p for p in printed)


def test_only_runs_the_named_tasks(tmp_path, paths):
    _, rows, _ = run_dev(tmp_path, paths, only=["D02"])
    assert [r["id"] for r in rows if r["type"] == "task"] == ["D02"]
    with pytest.raises(hs.TaskError, match="not dev tasks: D09"):
        run_dev(tmp_path, paths, only=["D09"])


def test_regrade_reproduces_the_run(tmp_path, paths):
    path, rows, _ = run_dev(tmp_path, paths)
    out_path = hs.regrade(path, echo=lambda s: None)
    with open(out_path, encoding="utf-8") as f:
        again = [json.loads(line) for line in f]
    assert again[0]["regraded_from"] == os.path.basename(path)
    assert [r["checks"] for r in again[1:-1]] == [r["checks"] for r in rows[1:-1]]
    assert again[-1]["passed"] == rows[-1]["passed"]


def test_show_prints_one_task(tmp_path, paths):
    path, _, _ = run_dev(tmp_path, paths)
    printed = []
    hs.show(path, "D01", echo=printed.append)
    text = "\n".join(printed)
    assert "Question: Question D01" in text and "population ok" in text and "rows_1:" in text


def test_a_harness_error_is_logged_and_the_run_goes_on(tmp_path, paths, monkeypatch):
    real = hs.gr.answer

    def flaky(question, sender, **kw):
        if question == "Question D01":
            raise RuntimeError("boom")
        return real(question, sender, **kw)
    monkeypatch.setattr(hs.gr, "answer", flaky)
    _, rows, _ = run_dev(tmp_path, paths)
    assert rows[1]["outputs"]["status"] == "harness_error"
    assert rows[1]["outputs"]["stop_reason"] == "RuntimeError" and not rows[1]["pass"]
    assert rows[2]["pass"]


# ---------- references ----------

@pytest.mark.parametrize("fields, message", [
    ({"figures": "SELECT nope FROM complaints_v"}, "no such column"),
    ({"figures": "SELECT product, NULL AS n FROM complaints_v GROUP BY product"}, "empty"),
    ({"figures": "SELECT complaint_id FROM complaints_v WHERE 0"}, "no rows"),
    ({"figures": "SELECT a.complaint_id FROM complaints_v a, complaints_v b"}, "more than 12"),
    ({"figures": "SELECT 5 AS _n"}, "nothing is graded"),
    ({"figures": "SELECT -3 AS change", "direction": "up", "direction_of": "change"},
     "disagrees"),
    ({"figures": "SELECT 3 AS change", "direction": "up", "direction_of": "delta"},
     "no column delta"),
    ({"population": "SELECT product FROM complaints_v", "causes": 1}, "complaint_id"),
    ({"population": "SELECT complaint_id FROM complaints_v WHERE 0", "causes": 1},
     "no narratives"),
])
def test_bad_references_stop_the_run_before_any_model_call(tmp_path, paths, fields, message):
    tp = write(tmp_path, task("D01", **fields))
    calls = []

    def counting(task):
        calls.append(task["id"])
        return planner_for(task)
    agent_kw = {k: paths[k] for k in ("agent_path", "layer_path")}
    with pytest.raises(hs.TaskError, match=message):
        hs.run("dev", tasks_path=tp, runs_dir=str(tmp_path / "runs"), planner_for=counting,
               agent_kw=agent_kw)
    assert calls == []


def test_check_prints_references_and_warns_on_small_numbers(tmp_path, paths):
    printed = []
    hs.check(tasks_for_run(tmp_path), agent_path=paths["agent_path"],
             layer_path=paths["layer_path"], echo=printed.append)
    text = "\n".join(printed)
    assert "figures: n=5" in text and "small number" in text
    assert "8 drawn; top 1 causes graded" in text and "(the test run needs 12 + 3)" in text


# ---------- the test run's guards ----------

def fifteen(tmp_path):
    body = "".join(task(f"N{i:02d}", split="test", figures=CARD_COUNT) for i in range(1, 13))
    body += "".join(task(f"A{i:02d}", split="test", kind="clarify") for i in range(1, 4))
    return write(tmp_path, body)


def test_test_needs_final(tmp_path):
    with pytest.raises(hs.TaskError, match="add --final"):
        hs.run("test", tasks_path=fifteen(tmp_path), runs_dir=str(tmp_path / "runs"))


def test_final_is_only_for_test(tmp_path):
    with pytest.raises(hs.TaskError, match="only for the test split"):
        hs.run("dev", final=True, tasks_path=tasks_for_run(tmp_path),
               runs_dir=str(tmp_path / "runs"))


def test_final_needs_twelve_and_three(tmp_path):
    tp = write(tmp_path, task("N01", split="test", figures=CARD_COUNT))
    with pytest.raises(hs.TaskError, match="found 1 and 0"):
        hs.run("test", final=True, tasks_path=tp, runs_dir=str(tmp_path / "runs"))


def test_final_runs_once(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "agent-test-20261201T000000Z.jsonl").write_text("{}\n")
    with pytest.raises(hs.TaskError, match="run once"):
        hs.run("test", final=True, tasks_path=fifteen(tmp_path), runs_dir=str(runs))


def test_final_refuses_uncommitted_code(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "git_commit", lambda: "726d131-dirty")
    with pytest.raises(hs.TaskError, match="commit every change first"):
        hs.run("test", final=True, tasks_path=fifteen(tmp_path), runs_dir=str(tmp_path / "runs"))


def test_final_covers_every_test_task(tmp_path):
    with pytest.raises(hs.TaskError, match="drop --only"):
        hs.run("test", final=True, only=["N01"], tasks_path=fifteen(tmp_path),
               runs_dir=str(tmp_path / "runs"))


def test_bedrock_runs_need_aws_credentials(tmp_path, paths, monkeypatch):
    monkeypatch.setattr(hs, "aws_ready", lambda: "no AWS credentials found: run aws login first")
    agent_kw = {k: paths[k] for k in ("agent_path", "layer_path")}
    with pytest.raises(hs.TaskError, match="aws login"):
        hs.run("dev", tasks_path=tasks_for_run(tmp_path), runs_dir=str(tmp_path / "runs"),
               agent_kw=agent_kw)
    assert not os.path.exists(tmp_path / "runs")


def test_main_prints_one_line_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hs, "TASKS_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(hs, "check", lambda: hs.load_tasks(hs.TASKS_PATH))
    assert hs.main(["harness", "check"]) == 1
    assert capsys.readouterr().out.startswith("error:")
    assert hs.main(["harness", "nonsense"]) == 2
