"""The LangGraph agent with a scripted planner in place of Bedrock, on the fake database."""
import hashlib
import shutil
from typing import Any, List

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import graph as gr
from agent import report as rp
from agent import semantic as sem
from agent import send as sd
from tests.test_report import PLANTED, ROWS, FakeBedrock, FakeSes, AwsError, make_source

# Pinned per plan version: an edit to the instructions or the layer description
# fails here until PLAN_VERSION is bumped.
PROMPT_SHA256 = {
    "plan-1": "00c179e76ce0013722343da75da2ba68c71737473253fda5ad6a68520d697608",
    "plan-2": "716519be9b23597361146be22cb0143e13d4e1a393dbda735a0612fc7726353d",
}

BY_PRODUCT = ("SELECT product, COUNT(*) AS complaints FROM complaints_v GROUP BY product "
              "ORDER BY product")
FINDINGS = ["Fraud claims are the most common root cause at 42.9%.",
            "Each product has 5 complaints in the snapshot."]


def call(name, n=1, **args):
    return {"name": name, "args": args, "id": f"{name}-{n}"}


def turn(*calls, text=""):
    return AIMessage(content=text, tool_calls=list(calls),
                     usage_metadata={"input_tokens": 100, "output_tokens": 10,
                                     "total_tokens": 110})


class ScriptedPlanner(BaseChatModel):
    """Plays a list of AIMessages (or functions of the messages seen) in order.
    After the script, repeats `after` if given, else ends with plain text."""
    script: List[Any] = []
    after: Any = None
    seen: List[Any] = []
    bound: List[str] = []
    fail: Any = None

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        self.bound = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.fail:
            raise self.fail
        self.seen.append(list(messages))
        if self.script:
            msg = self.script.pop(0)
        elif self.after is not None:
            msg = self.after
        else:
            msg = turn(text="Done.")
        if callable(msg):
            msg = msg(messages)
        n = len(self.seen)                    # fresh ids: add_messages replaces repeats
        msg = AIMessage(content=msg.content, usage_metadata=msg.usage_metadata,
                        tool_calls=[{**c, "id": f"{c['id']}-t{n}"} for c in msg.tool_calls])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def paths(tmp_path):
    src = str(tmp_path / "complaints.sqlite")
    make_source(src)
    layer = str(tmp_path / "semantic.toml")
    shutil.copy(sem.LAYER_PATH, layer)
    agent = str(tmp_path / "agent.sqlite")
    sem.build(source_path=src, agent_path=agent, layer_path=layer)
    return {"agent_path": agent, "layer_path": layer, "tag_client": FakeBedrock(),
            "tag_cache": str(tmp_path / "cache.sqlite"), "tag_sleep": lambda s: None}


def ask(paths, planner, sender=None, question="What are the root causes of card and "
        "checking complaints?"):
    sender = sender or sd.DryRunSender(to="fixed@example.com")
    result, run = gr.answer(question, sender, planner=planner, **paths)
    if run:
        run.close()
    return result, sender


def happy_script():
    return [
        turn(call("run_sql", sql=BY_PRODUCT)),
        turn(call("tag_root_causes", sql="SELECT complaint_id FROM complaints_v"),
             call("clean_and_chart", handle="rows_1", title="Complaints by product",
                  x="product", y="complaints")),
        turn(call("clean_and_chart", 2, handle="tags_1", title="Root causes")),
        turn(call("build_report", title="Root causes of complaints", findings=FINDINGS,
                  charts=["chart_1", "chart_2"], tags="tags_1")),
        turn(text="The report is ready."),
    ]


# ---------- the whole run ----------

def test_a_run_builds_and_sends_once(paths):
    planner = ScriptedPlanner(script=happy_script())
    result, sender = ask(paths, planner)
    assert result["status"] == "sent" and result["stop_reason"] is None
    assert len(sender.sent) == 1 and sender.sent[0]["to"] == "fixed@example.com"
    assert sender.sent[0]["attachments"][0]["sha256"] == result["report"]["sha256"]
    assert result["counts"] == {"turns": 5, "tool_calls": 5, "tag_runs": 1, "failed_calls": 0}
    tools = [s["tool"] for s in result["tool_trace"]]
    assert tools[0] == "run_sql" and tools[3:] == ["clean_and_chart", "build_report"]
    assert sorted(tools[1:3]) == ["clean_and_chart", "tag_root_causes"]   # one turn, any order
    assert result["report"]["quotes"] and result["report"]["charts"] == ["chart_1", "chart_2"]
    assert result["final_text"] == "The report is ready."
    assert result["tokens"] == {"input": 500, "output": 50}
    assert result["queries"] == {"rows_1": BY_PRODUCT}
    assert result["tag_queries"] == {"tags_1": "SELECT complaint_id FROM complaints_v"}
    assert planner.bound == list(gr.TOOL_NAMES)                     # no send tool


def test_the_planner_never_sees_narrative_text(paths):
    planner = ScriptedPlanner(script=happy_script())
    ask(paths, planner)
    seen = "\n".join(str(m.content) for msgs in planner.seen for m in msgs)
    for row in ROWS.values():
        if row[2]:
            assert row[2][:30] not in seen
    assert PLANTED not in seen
    assert "Example complaints" not in seen


def test_system_prompt_holds_the_tables_and_is_pinned(paths):
    planner = ScriptedPlanner(script=[turn(text="Nothing to do.")])
    ask(paths, planner)
    system = planner.seen[0][0].content
    assert "Table complaints_v" in system and "text_has" in system
    real = gr.system_prompt(sem.load())
    assert hashlib.sha256(real.encode()).hexdigest() == PROMPT_SHA256[gr.PLAN_VERSION]


def test_sql_errors_go_back_to_the_model_which_corrects_them(paths):
    script = happy_script()
    script.insert(0, turn(call("run_sql", 0, sql="SELECT nope FROM complaints_v")))
    planner = ScriptedPlanner(script=script)
    result, sender = ask(paths, planner)
    errors = [m for m in planner.seen[1] if isinstance(m, ToolMessage)]
    assert errors[0].content.startswith("ERROR: SQLite error: no such column: nope")
    assert result["status"] == "sent" and result["counts"]["failed_calls"] == 1
    assert result["tool_trace"][0]["error"].startswith("SQLite error: no such column: nope")


def test_parallel_calls_share_one_connection_safely(paths):
    sqls = [f"SELECT COUNT(*) AS n{i} FROM complaints_v" for i in range(5)]
    planner = ScriptedPlanner(script=[turn(*[call("run_sql", i, sql=s)
                                             for i, s in enumerate(sqls)])])
    result, _ = ask(paths, planner)
    assert result["counts"]["failed_calls"] == 0 and len(result["queries"]) == 5


# ---------- ways a run ends without an email ----------

def test_a_clarifying_question_ends_the_run(paths):
    planner = ScriptedPlanner(script=[
        turn(call("ask_clarifying_question", question="Which **year** do you mean?\nOnly `2026` "
                                                       "is available.")),
        turn(text="should never be asked")])
    result, sender = ask(paths, planner)
    assert result["status"] == "needs_clarification"
    assert result["clarifying_question"] == "Which year do you mean? Only 2026 is available."
    assert result["report"] is None and sender.sent == [] and len(planner.seen) == 1


def test_no_report_means_no_email(paths):
    planner = ScriptedPlanner(script=[turn(call("run_sql", sql=BY_PRODUCT)),
                                      turn(text="Here is the answer in words.")])
    result, sender = ask(paths, planner)
    assert result["status"] == "no_report" and sender.sent == []


def test_turn_limit_stops_the_run_even_with_a_report_built(paths):
    script = happy_script()[:4]
    planner = ScriptedPlanner(script=script, after=turn(call("run_sql", sql="SELECT 1 AS x")))
    result, sender = ask(paths, planner)
    assert result["status"] == "stopped" and result["stop_reason"] == "turn_limit"
    assert result["counts"]["turns"] == gr.MAX_TURNS and len(planner.seen) == gr.MAX_TURNS
    assert result["report"] is not None and sender.sent == []


def test_tool_call_limit(paths):
    calls = [call("run_sql", i, sql=f"SELECT {i} AS x") for i in range(gr.MAX_TOOL_CALLS + 5)]
    planner = ScriptedPlanner(script=[turn(*calls)])
    result, sender = ask(paths, planner)
    assert result["stop_reason"] == "tool_call_limit"
    assert result["counts"]["tool_calls"] == gr.MAX_TOOL_CALLS
    assert len(result["queries"]) == gr.MAX_TOOL_CALLS and sender.sent == []


def test_failures_in_a_row_stop_the_run(paths):
    planner = ScriptedPlanner(script=[], after=turn(call("run_sql", sql="SELECT nope")))
    result, _ = ask(paths, planner)
    assert result["stop_reason"] == "failures_in_a_row"
    assert result["counts"]["failed_calls"] == gr.MAX_FAILS_IN_A_ROW


def test_a_success_resets_the_failure_count(paths):
    bad, good = turn(call("run_sql", sql="SELECT nope")), turn(call("run_sql", sql="SELECT 1"))
    planner = ScriptedPlanner(script=[bad, bad, bad, good, bad, bad, bad,
                                      turn(text="Giving up.")])
    result, _ = ask(paths, planner)
    assert result["stop_reason"] is None and result["status"] == "no_report"
    assert result["counts"]["failed_calls"] == 6


def test_at_most_two_tag_runs(paths):
    tag = lambda n: call("tag_root_causes", n, sql="SELECT complaint_id FROM narratives_v")
    planner = ScriptedPlanner(script=[turn(tag(1)), turn(tag(2)), turn(tag(3))])
    result, _ = ask(paths, planner)
    assert result["counts"]["tag_runs"] == 2
    assert result["tool_trace"][2]["error_kind"] == "limit"
    msgs = [m for m in planner.seen[3] if isinstance(m, ToolMessage)]
    assert "At most 2 tag_root_causes calls" in msgs[-1].content


def test_a_model_error_keeps_only_the_code(paths):
    planner = ScriptedPlanner(fail=AwsError("ThrottlingException"))
    result, sender = ask(paths, planner)
    assert result["status"] == "stopped" and result["stop_reason"] == "model_error"
    assert result["model_error"] == "ThrottlingException" and sender.sent == []


def test_a_failed_send_is_reported(paths):
    planner = ScriptedPlanner(script=happy_script())
    ses = sd.SesSender(to="fixed@example.com", sender="agent@example.com",
                       client=FakeSes(fail=AwsError("MessageRejected")))
    result, _ = ask(paths, planner, sender=ses)
    assert result["status"] == "send_failed" and result["sent"]["error"] == "MessageRejected"


@pytest.mark.parametrize("question", ["", "   ", "x" * (gr.MAX_QUESTION + 1)])
def test_bad_questions_are_refused_before_any_model_call(paths, question):
    planner = ScriptedPlanner(script=[])
    result, _ = ask(paths, planner, question=question)
    assert result["status"] == "invalid_question" and planner.seen == []


def test_final_text_from_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "Report\n ready."},
                             {"type": "tool_use", "id": "x", "name": "y", "input": {}}])
    assert gr.text_of(msg) == "Report ready."


def test_the_real_planner_sends_the_expected_converse_request(paths, monkeypatch):
    """bedrock_planner() through the graph, with botocore's Stubber in place of AWS."""
    import boto3
    from botocore.stub import Stubber
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    client = boto3.client("bedrock-runtime", region_name=gr.REGION)
    sent = []
    client.meta.events.register("provide-client-params.bedrock-runtime.Converse",
                                lambda params, **kw: sent.append(params))
    stub = Stubber(client)
    stub.add_response("converse", {
        "output": {"message": {"role": "assistant", "content": [{"text": "Nothing to do."}]}},
        "stopReason": "end_turn", "usage": {"inputTokens": 7, "outputTokens": 3,
                                             "totalTokens": 10}, "metrics": {"latencyMs": 1}})
    stub.activate()
    monkeypatch.setattr(gr.tg, "bedrock_client", lambda: client)
    result, run = gr.answer("How many complaints?", sd.DryRunSender(), **paths)
    run.close()
    (req,) = sent
    assert req["modelId"] == "global.anthropic.claude-sonnet-4-6"
    assert req["inferenceConfig"] == {"maxTokens": gr.MAX_TOKENS, "temperature": 0.0}
    assert req["system"][0]["text"] == gr.system_prompt(sem.load(paths["layer_path"]))
    specs = [t["toolSpec"] for t in req["toolConfig"]["tools"]]
    assert [s["name"] for s in specs] == list(gr.TOOL_NAMES)
    assert all(len(s["description"]) > 40 for s in specs)
    assert result["status"] == "no_report" and result["tokens"] == {"input": 7, "output": 3}
    assert result["final_text"] == "Nothing to do."


def test_numbers_typed_into_sql_do_not_pass_the_check(paths):
    """The S6 real run: refused for 30.2%, the planner typed 15.6 + 14.6 into SQL."""
    script = happy_script()
    script[3:4] = [
        turn(call("build_report", title="Root causes", findings=["Together 44.0% of tagged "
             "narratives."], charts=["chart_1"], tags="tags_1")),
        turn(call("run_sql", 9, sql="SELECT COUNT(*) AS n, ROUND(42.9 + 1.1, 1) AS together "
                                    "FROM complaints_v")),
        turn(call("build_report", 2, title="Root causes", findings=["Together 44.0% of "
             "tagged narratives."], charts=["chart_1"], tags="tags_1")),
        turn(call("build_report", 3, title="Root causes of complaints", findings=FINDINGS,
                  charts=["chart_1", "chart_2"], tags="tags_1")),
    ]
    planner = ScriptedPlanner(script=script)
    result, sender = ask(paths, planner)
    reports = [s for s in result["tool_trace"] if s["tool"] == "build_report"]
    assert [s["ok"] for s in reports] == [False, False, True]
    assert all("44.0% not found" in s["error"] for s in reports[:2])
    told = [m.content for m in planner.seen[5] if isinstance(m, ToolMessage)][-1]
    assert "Column(s) together don't depend on the data" in told
    assert result["status"] == "sent" and len(sender.sent) == 1
