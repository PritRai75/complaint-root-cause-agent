"""Tests for the deployment (session 10): the agent's FastAPI app, the submit and
status functions, the package builder and the SAM template. Fakes only: no AWS,
no network, no databases."""
import io
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent import graph
from agent import tagger as tg
from deploy import agent_app as aa
from deploy import package as pk
from deploy.front import handlers as fh

ROOT = Path(__file__).resolve().parents[1]
RID = "run-" + "a" * 32
RID2 = "run-" + "b" * 32


# ---------- fakes ----------

class FakeS3:
    """Enough of S3 for the app and the functions: objects in a dict."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            err = Exception("missing")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys]} if keys else {"KeyCount": 0}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://signed.example/{Params['Key']}?expires={ExpiresIn}"

    def json(self, key, bucket="results"):
        return json.loads(self.objects[(bucket, key)])


class FakeAgentCore:
    def __init__(self, answer=b'{"status": "started"}', error=None):
        self.answer, self.error, self.calls = answer, error, []

    def invoke_agent_runtime(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return {"response": io.BytesIO(self.answer)}


class Clients:
    def __init__(self, s3=None, agentcore=None):
        self.s3, self.agentcore = s3 or FakeS3(), agentcore or FakeAgentCore()


class FakeRun:
    def __init__(self, report=None):
        self.store = type("S", (), {"report": report})()
        self.closed = False

    def close(self):
        self.closed = True


def client_error(code):
    err = Exception(f"boom {code} with the question echoed")
    err.response = {"Error": {"Code": code, "Message": "secret question text"}}
    return err


ENV = {"RESULTS_BUCKET": "results", "AGENT_RUNTIME_ARN": "arn:aws:bedrock-agentcore:x:1:runtime/a",
       "DAILY_RUNS": "3"}
T0 = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)


def post(question=None, body=None):
    if body is None:
        body = json.dumps({"question": question})
    return {"body": body}


def endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


# ---------- the agent app ----------

def make_app(started=None, results=None):
    started = [] if started is None else started
    health = aa.Health(clock=lambda: 1000)
    results = results or aa.Results(bucket="results", client=FakeS3())
    app = aa.create_app(health=health, results_factory=lambda: results,
                        start=lambda target, args: started.append(args), tag_cache="/tmp/x")
    return app, health, started, results


def test_ping_is_healthy_then_busy_and_stamps_only_changes():
    clock = iter([100, 200, 300, 400])
    h = aa.Health(clock=lambda: next(clock))
    assert h.status() == {"status": "Healthy", "time_of_last_update": 100}
    assert h.status()["time_of_last_update"] == 100
    assert h.start(RID)
    assert h.status() == {"status": "HealthyBusy", "time_of_last_update": 200}
    assert not h.start(RID2)
    assert h.status()["time_of_last_update"] == 200
    h.end(RID)
    assert h.status() == {"status": "Healthy", "time_of_last_update": 300}


def test_invocations_starts_one_run_in_the_background():
    app, health, started, _ = make_app()
    out = endpoint(app, "/invocations")(body={"run_id": RID, "question": "How many?"},
                                        session_id=RID)
    assert out == {"status": "started", "run_id": RID}
    assert len(started) == 1 and started[0][0] == RID and started[0][1] == "How many?"
    assert started[0][6] == "/tmp/x"
    assert endpoint(app, "/ping")()["status"] == "HealthyBusy"
    second = endpoint(app, "/invocations")(body={"run_id": RID2, "question": "q"},
                                           session_id=None)
    assert second.status_code == 409 and len(started) == 1


@pytest.mark.parametrize("body,session,why", [
    ({"run_id": "run-123", "question": "q"}, None, "run_id"),
    ({"run_id": RID.upper(), "question": "q"}, None, "run_id"),
    ({"question": "q"}, None, "run_id"),
    ({"run_id": RID, "question": "q"}, RID2, "session"),
    ({"run_id": RID, "question": "   "}, None, "question"),
    ({"run_id": RID, "question": 5}, None, "question"),
    ({"run_id": RID, "question": "a" * 1001}, None, "1,000"),
])
def test_invocations_refuses_bad_input(body, session, why):
    app, health, started, _ = make_app()
    out = endpoint(app, "/invocations")(body=body, session_id=session)
    assert out.status_code == 400 and why in json.loads(out.body)["error"]
    assert started == [] and health.status()["status"] == "Healthy"


def test_question_limit_matches_the_graph():
    assert aa.MAX_QUESTION == graph.MAX_QUESTION == fh.MAX_QUESTION


def test_run_job_writes_status_report_and_result_and_closes():
    s3 = FakeS3()
    results = aa.Results(bucket="results", client=s3)
    health = aa.Health(clock=lambda: 1)
    health.start(RID)
    run = FakeRun(report={"bytes": b"DOCX", "filename": "r.docx"})
    result = {"status": "sent", "question": "q", "report": {"title": "T", "findings": ["f"]},
              "counts": {"turns": 3, "tool_calls": 4, "tag_runs": 0, "failed_calls": 0},
              "tokens": {"input": 10, "output": 2}, "tool_trace": [{"tool": "run_sql"}],
              "sent": {"sender": "ses", "message_id": "m1", "subject": "Complaint report: T",
                       "attachment_sha256": "x"}}
    seen = {}

    def answer(question, sender, tag_cache):
        seen.update(question=question, sender=sender, cache=tag_cache)
        return result, run
    aa.run_job(RID, "q", results, health, answer=answer, sender_factory=lambda: "SENDER",
               tag_cache="/tmp/c", clock=iter([0.0, 12.0, 12.0, 12.0]).__next__)
    assert seen == {"question": "q", "sender": "SENDER", "cache": "/tmp/c"}
    assert s3.objects[("results", f"runs/{RID}/report.docx")] == b"DOCX"
    st = s3.json(f"runs/{RID}/status.json")
    assert st["status"] == "sent" and st["has_report"] is True and st["seconds"] == 12.0
    res = s3.json(f"runs/{RID}/result.json")
    assert res["run_id"] == RID and res["report"]["title"] == "T"
    assert "tool_trace" not in res and res["sent"] == {"sender": "ses", "message_id": "m1"}
    assert run.closed and health.status()["status"] == "Healthy"


def test_run_job_records_an_error_code_only(capsys):
    s3 = FakeS3()
    results = aa.Results(bucket="results", client=s3)
    health = aa.Health(clock=lambda: 1)
    health.start(RID)

    def answer(question, sender, tag_cache):
        raise client_error("ThrottlingException")
    aa.run_job(RID, "my secret question", results, health, answer=answer,
               sender_factory=lambda: None, clock=lambda: 0.0)
    st = s3.json(f"runs/{RID}/status.json")
    assert st["status"] == "error" and st["error"] == "ThrottlingException"
    logs = capsys.readouterr().out
    assert "secret" not in logs and "ThrottlingException" in logs
    assert health.status()["status"] == "Healthy"


def test_run_job_without_a_run_writes_no_report():
    s3 = FakeS3()
    results = aa.Results(bucket="results", client=s3)
    health = aa.Health(clock=lambda: 1)
    health.start(RID)
    aa.run_job(RID, "q", results, health,
               answer=lambda q, s, c: ({"status": "invalid_question", "error": "x"}, None),
               sender_factory=lambda: None, clock=lambda: 0.0)
    assert s3.json(f"runs/{RID}/status.json")["status"] == "invalid_question"
    assert ("results", f"runs/{RID}/report.docx") not in s3.objects


def test_logs_never_hold_the_question(capsys):
    app, health, started, results = make_app()
    endpoint(app, "/invocations")(body={"run_id": RID, "question": "zebra question"},
                                  session_id=None)
    aa.run_job(RID, "zebra question", results, health,
               answer=lambda q, s, c: ({"status": "needs_clarification",
                                        "clarifying_question": "zebra?"}, FakeRun()),
               sender_factory=lambda: None, clock=lambda: 0.0)
    assert "zebra" not in capsys.readouterr().out


def test_prepare_copies_the_tag_cache_and_sets_matplotlib(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tag_cache.sqlite").write_bytes(b"cache")
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.chdir(ROOT)
    cache = aa.prepare(tmp_path, tmp=str(tmp_path / "tmp"))
    assert Path(cache).read_bytes() == b"cache"
    assert Path.cwd() == tmp_path
    assert aa.os.environ["MPLCONFIGDIR"].startswith(str(tmp_path / "tmp"))


# ---------- submit ----------

def test_submit_starts_a_run_with_the_run_id_as_session():
    c = Clients()
    out = fh.submit(post("How many card complaints?"), None, clients=c, environ=ENV,
                    clock=lambda: T0, new_id=lambda: RID)
    assert out["statusCode"] == 202
    assert json.loads(out["body"]) == {"run_id": RID, "status": "queued", "poll": f"/runs/{RID}"}
    call = c.agentcore.calls[0]
    assert call["runtimeSessionId"] == RID and call["agentRuntimeArn"] == ENV["AGENT_RUNTIME_ARN"]
    assert json.loads(call["payload"]) == {"run_id": RID, "question": "How many card complaints?"}
    assert c.s3.json(f"runs/{RID}/status.json")["status"] == "queued"
    assert ("results", f"days/2026-11-16/{RID}") in c.s3.objects
    assert len(RID) >= 33           # AgentCore's minimum session ID length


@pytest.mark.parametrize("event,why", [
    ({}, "JSON body"),
    ({"body": "not json"}, "valid JSON"),
    ({"body": "[1]"}, "non-empty"),
    ({"body": json.dumps({"question": ""})}, "non-empty"),
    ({"body": json.dumps({"question": "a" * 1001})}, "1,000"),
    ({"body": "x" * 8001}, "8,000"),
    ({"body": "e30=", "isBase64Encoded": True}, "plain JSON"),
])
def test_submit_refuses_bad_bodies(event, why):
    c = Clients()
    out = fh.submit(event, None, clients=c, environ=ENV, clock=lambda: T0)
    assert out["statusCode"] == 400 and why in json.loads(out["body"])["error"]
    assert c.agentcore.calls == [] and c.s3.objects == {}


def test_submit_normalises_whitespace():
    c = Clients()
    fh.submit(post("  How   many\n complaints? "), None, clients=c, environ=ENV,
              clock=lambda: T0, new_id=lambda: RID)
    assert json.loads(c.agentcore.calls[0]["payload"])["question"] == "How many complaints?"


def done_run(s3, run_id, day="2026-11-16", status="sent", started=None):
    s3.put_object(Bucket="results", Key=f"days/{day}/{run_id}", Body=b"")
    st = {"run_id": run_id, "status": status}
    if started:
        st["started_at"] = started
    s3.put_object(Bucket="results", Key=f"runs/{run_id}/status.json", Body=json.dumps(st))


def test_submit_enforces_the_daily_cap():
    c = Clients()
    for i in range(3):
        done_run(c.s3, f"run-{i:032x}")
    out = fh.submit(post("q"), None, clients=c, environ=ENV, clock=lambda: T0)
    assert out["statusCode"] == 429 and "daily limit of 3" in json.loads(out["body"])["error"]
    assert c.agentcore.calls == []
    yesterday_only = Clients()
    for i in range(3):
        done_run(yesterday_only.s3, f"run-{i:032x}", day="2026-11-15")
    ok = fh.submit(post("q"), None, clients=yesterday_only, environ=ENV, clock=lambda: T0,
                   new_id=lambda: RID)
    assert ok["statusCode"] == 202


def test_submit_waits_for_an_active_run_but_not_a_stale_one():
    c = Clients()
    done_run(c.s3, RID2, day="2026-11-15", status="running",
             started=fh.iso(T0 - timedelta(minutes=10)))
    out = fh.submit(post("q"), None, clients=c, environ=ENV, clock=lambda: T0)
    assert out["statusCode"] == 429 and json.loads(out["body"])["active_run"] == RID2
    stale = Clients()
    done_run(stale.s3, RID2, status="running", started=fh.iso(T0 - timedelta(minutes=50)))
    ok = fh.submit(post("q"), None, clients=stale, environ=ENV, clock=lambda: T0,
                   new_id=lambda: RID)
    assert ok["statusCode"] == 202


def test_submit_records_a_failed_start_with_the_code_only(capsys):
    c = Clients(agentcore=FakeAgentCore(error=client_error("AccessDeniedException")))
    out = fh.submit(post("secret question"), None, clients=c, environ=ENV, clock=lambda: T0,
                    new_id=lambda: RID)
    assert out["statusCode"] == 502
    assert json.loads(out["body"])["code"] == "AccessDeniedException"
    st = c.s3.json(f"runs/{RID}/status.json")
    assert st["status"] == "error" and st["error"] == "AccessDeniedException"
    assert "secret" not in capsys.readouterr().out


def test_submit_treats_an_unexpected_agent_answer_as_a_failed_start():
    c = Clients(agentcore=FakeAgentCore(answer=b'{"status": "busy"}'))
    out = fh.submit(post("q"), None, clients=c, environ=ENV, clock=lambda: T0,
                    new_id=lambda: RID)
    assert out["statusCode"] == 502
    assert c.s3.json(f"runs/{RID}/status.json")["status"] == "error"


# ---------- status ----------

def test_status_checks_the_id_and_reports_missing_runs():
    c = Clients()
    bad = fh.status({"pathParameters": {"run_id": "run-xyz"}}, None, clients=c, environ=ENV)
    assert bad["statusCode"] == 400
    missing = fh.status({"pathParameters": {"run_id": RID}}, None, clients=c, environ=ENV)
    assert missing["statusCode"] == 404


def test_status_of_a_running_run_has_no_link():
    c = Clients()
    done_run(c.s3, RID, status="running", started=fh.iso(T0))
    out = json.loads(fh.status({"pathParameters": {"run_id": RID}}, None, clients=c,
                               environ=ENV)["body"])
    assert out["status"] == "running" and "report_url" not in out


def test_status_of_a_finished_run_links_the_report():
    c = Clients()
    c.s3.put_object(Bucket="results", Key=f"runs/{RID}/status.json", Body=json.dumps(
        {"run_id": RID, "status": "sent", "has_report": True, "seconds": 300.0}))
    c.s3.put_object(Bucket="results", Key=f"runs/{RID}/result.json", Body=json.dumps(
        {"question": "q", "report": {"title": "T", "findings": ["f1"], "filename": "r.docx",
                                     "bytes": 10, "sha256": "s", "quotes": [["fee", 1]]},
         "queries": {"rows_1": "SELECT 1"}}))
    out = json.loads(fh.status({"pathParameters": {"run_id": RID}}, None, clients=c,
                               environ=ENV)["body"])
    assert out["report"] == {"title": "T", "findings": ["f1"], "filename": "r.docx",
                             "bytes": 10, "sha256": "s"}
    assert out["report_url"].endswith(f"runs/{RID}/report.docx?expires=3600")
    assert out["question"] == "q" and "queries" not in out


def test_status_of_a_clarifying_question():
    c = Clients()
    c.s3.put_object(Bucket="results", Key=f"runs/{RID}/status.json", Body=json.dumps(
        {"run_id": RID, "status": "needs_clarification", "has_report": False}))
    c.s3.put_object(Bucket="results", Key=f"runs/{RID}/result.json", Body=json.dumps(
        {"clarifying_question": "Which measure?", "report": None}))
    out = json.loads(fh.status({"pathParameters": {"run_id": RID}}, None, clients=c,
                               environ=ENV)["body"])
    assert out["clarifying_question"] == "Which measure?" and "report_url" not in out


# ---------- the package ----------

def test_package_ships_the_code_the_agent_imports():
    files = pk.code_files()
    for needed in ("agent/graph.py", "agent/semantic.toml", "ingest/sources.py",
                   "ingest/build_db.py", "ingest/fetch.py", "eval/taxonomy.py",
                   "eval/taxonomy.toml", "deploy/agent_app.py"):
        assert needed in files
    assert not any(f.startswith(("tests/", "eval/labels", "eval/runs")) for f in files)


def test_package_refuses_the_crt():
    with pytest.raises(pk.PackageError, match="CRT"):
        pk.check_requirements("boto3==1\nbotocore[crt]==1  # laptop\n")
    pk.check_requirements("boto3==1\n# botocore[crt] is laptop only\n")


def test_platforms_include_glibc_2_28():
    assert "manylinux_2_28_aarch64" in pk.PLATFORMS and "manylinux2014_aarch64" in pk.PLATFORMS


def test_elf_check_finds_non_arm_libraries(tmp_path):
    def elf(machine):
        head = bytearray(b"\x7fELF\x02\x01\x01" + b"\x00" * 13)
        head[18:20] = machine.to_bytes(2, "little")
        return bytes(head)
    (tmp_path / "ok.so").write_bytes(elf(183))
    (tmp_path / "libx.so.1").write_bytes(elf(62))
    (tmp_path / "notelf.so").write_bytes(b"text")
    assert pk.elf_machine(tmp_path / "ok.so") == 183
    with pytest.raises(pk.PackageError, match="libx.so.1"):
        pk.check_binaries(tmp_path)


def fake_repo(tmp_path, cache=True):
    for rel in pk.code_files() + ["requirements.txt"]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    for rel in pk.DATA_FILES + (pk.OPTIONAL_DATA if cache else []):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"db " + rel.encode())
    return tmp_path


def fake_install(requirements, target):
    (target / "fastapi").mkdir()
    (target / "fastapi" / "__init__.py").write_text("x = 1\n")
    (target / "fastapi" / "__pycache__").mkdir()
    (target / "fastapi" / "__pycache__" / "x.pyc").write_bytes(b"pyc")


def test_build_makes_a_deterministic_zip_with_main_py(tmp_path):
    root = fake_repo(tmp_path)
    lines = []
    z1, m1 = pk.build(root, install=fake_install, check_data=lambda r: None, out=lines.append)
    first = z1.read_bytes()
    z2, m2 = pk.build(root, install=fake_install, check_data=lambda r: None, out=lines.append)
    assert z2.read_bytes() == first and m1["zip_sha256"] == m2["zip_sha256"]
    with zipfile.ZipFile(z1) as z:
        names = z.namelist()
        assert "main.py" in names and "deploy/agent_app.py" in names
        assert "data/agent.sqlite" in names and "data/tag_cache.sqlite" in names
        assert not any("__pycache__" in n for n in names)
        assert z.read("main.py").decode() == pk.MAIN_PY
        for info in z.infolist():
            mode = info.external_attr >> 16
            assert mode & 0o777 == (0o755 if info.is_dir() else 0o644)
            assert info.date_time == pk.FIXED_DATE
    assert (root / "build" / "agent.zip.sha256").read_text().startswith(m1["zip_sha256"])


def test_build_without_the_tag_cache_still_works(tmp_path):
    root = fake_repo(tmp_path, cache=False)
    z, m = pk.build(root, install=fake_install, check_data=lambda r: None, out=lambda s: None)
    assert "data/tag_cache.sqlite" not in m["files"]


def test_build_enforces_the_size_limits(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(pk.PackageError, match="unzipped size"):
        pk.build(root, install=fake_install, check_data=lambda r: None, max_unzipped=10,
                 out=lambda s: None)
    with pytest.raises(pk.PackageError, match="zip size"):
        pk.build(root, install=fake_install, check_data=lambda r: None, max_zipped=10,
                 out=lambda s: None)


def test_build_needs_the_databases(tmp_path):
    root = fake_repo(tmp_path)
    (root / "data" / "agent_text.sqlite").unlink()
    with pytest.raises(pk.PackageError, match="agent_text.sqlite"):
        pk.build(root, install=fake_install, check_data=lambda r: None, out=lambda s: None)


def test_upload_names_the_zip_by_its_hash_and_skips_a_repeat(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "agent.zip").write_bytes(b"zipbytes")

    class Cfn:
        def describe_stacks(self, StackName):
            assert StackName == "aws-sam-cli-managed-default"
            return {"Stacks": [{"Outputs": [{"OutputKey": "SourceBucket",
                                             "OutputValue": "sam-bucket"}]}]}

    class S3:
        def __init__(self):
            self.uploaded = []

        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in self.uploaded:
                raise client_error("404")

        def upload_file(self, path, bucket, key):
            self.uploaded.append((bucket, key))
    s3 = S3()
    bucket, key = pk.upload(tmp_path, cfn=Cfn(), s3=s3, out=lambda s: None)
    assert bucket == "sam-bucket" and re.match(r"^complaint-agent/agent-[0-9a-f]{16}\.zip$", key)
    assert (tmp_path / "build" / "code-key.txt").read_text() == key
    lines = []
    pk.upload(tmp_path, cfn=Cfn(), s3=s3, out=lines.append)
    assert len(s3.uploaded) == 1 and lines[0].startswith("Already uploaded")


# ---------- the template ----------

class CfnLoader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {suffix: value}


CfnLoader.add_multi_constructor("!", _tag)


@pytest.fixture(scope="module")
def template():
    return yaml.load((ROOT / "template.yaml").read_text(encoding="utf-8"), Loader=CfnLoader)


def statements(template, role):
    return [s for p in template["Resources"][role]["Properties"]["Policies"]
            for s in p["PolicyDocument"]["Statement"]]


def test_template_model_matches_the_code(template):
    assert "global." + template["Parameters"]["PlannerModel"]["Default"] == tg.MODEL_ID


def test_template_runtime_runs_main_py_on_python_3_13(template):
    rt = template["Resources"]["AgentRuntime"]["Properties"]
    code = rt["AgentRuntimeArtifact"]["CodeConfiguration"]
    assert code["Runtime"] == "PYTHON_3_13" and code["EntryPoint"] == ["main.py"]
    assert rt["LifecycleConfiguration"]["MaxLifetime"] <= 3600
    assert set(rt["EnvironmentVariables"]) == {"RESULTS_BUCKET", "REPORT_TO", "REPORT_FROM"}
    assert re.match(template["Parameters"]["CodeKey"]["AllowedPattern"],
                    "complaint-agent/agent-0123456789abcdef.zip")


def test_template_has_no_wildcard_actions_or_open_resources(template):
    for role in ("AgentRole", "SubmitRole", "StatusRole"):
        for s in statements(template, role):
            actions = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
            assert all("*" not in a for a in actions), (role, s["Sid"])
            resources = s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
            assert all(r != "*" for r in resources), (role, s["Sid"])


def test_template_agent_role_writes_only_run_files_and_sends_one_way(template):
    by_sid = {s["Sid"]: s for s in statements(template, "AgentRole")}
    assert by_sid["WriteRunFiles"]["Action"] == "s3:PutObject"
    assert by_sid["WriteRunFiles"]["Resource"] == {"Sub": "${ResultsBucket.Arn}/runs/*"}
    assert by_sid["SendOneWay"]["Action"] == "ses:SendEmail"
    assert len(by_sid["SendOneWay"]["Resource"]) == 2
    assert not any("s3:GetObject" in str(s["Action"]) and "ResultsBucket" in str(s["Resource"])
                   for s in by_sid.values())


def test_template_status_role_can_tell_missing_from_forbidden(template):
    """S3 returns NoSuchKey for a missing object only to callers with s3:ListBucket;
    without it, an unknown run ID came back as AccessDenied and the status function
    crashed (502) in the S10 smoke check."""
    by_sid = {s["Sid"]: s for s in statements(template, "StatusRole")}
    assert by_sid["MissingMeansNotFound"]["Action"] == "s3:ListBucket"
    assert by_sid["MissingMeansNotFound"]["Resource"] == {"GetAtt": "ResultsBucket.Arn"}
    assert "Condition" not in by_sid["MissingMeansNotFound"]


def test_status_crashes_rather_than_hiding_access_denied():
    """AccessDenied is not read as 'no such run': a real permission error must show."""
    s3 = FakeS3()

    def denied(Bucket, Key):
        raise client_error("AccessDenied")
    s3.get_object = denied
    with pytest.raises(Exception):
        fh.status({"pathParameters": {"run_id": RID}}, None, clients=Clients(s3=s3), environ=ENV)


def test_template_api_needs_a_key_and_caps_runs(template):
    api = template["Resources"]["AgentApi"]["Properties"]
    assert api["Auth"]["ApiKeyRequired"] is True
    assert api["Auth"]["UsagePlan"]["Throttle"] == {"RateLimit": 1, "BurstLimit": 2}
    assert template["Parameters"]["DailyRuns"]["Default"] == 3
    env = template["Resources"]["SubmitFunction"]["Properties"]["Environment"]["Variables"]
    assert env["DAILY_RUNS"] == {"Ref": "DailyRuns"}


def test_template_bucket_is_private_and_expires_runs(template):
    b = template["Resources"]["ResultsBucket"]["Properties"]
    assert all(b["PublicAccessBlockConfiguration"].values())
    rules = {r["Prefix"]: r["ExpirationInDays"] for r in b["LifecycleConfiguration"]["Rules"]}
    assert rules == {"runs/": 30, "days/": 7}


def test_front_requirements_pin_the_same_boto3():
    pins = dict(l.split("==") for l in (ROOT / "requirements.txt").read_text().split()
                if "==" in l)
    front = dict(l.split("==") for l in
                 (ROOT / "deploy/front/requirements.txt").read_text().split() if "==" in l)
    assert front == {"boto3": pins["boto3"]}


def test_samconfig_names_the_stack_the_smoke_test_reads():
    import tomllib
    from deploy import smoke
    cfg = tomllib.loads((ROOT / "samconfig.toml").read_text(encoding="utf-8"))
    g = cfg["default"]["global"]["parameters"]
    assert g["stack_name"] == smoke.STACK and g["region"] == smoke.REGION == tg.REGION
    assert pk.KEY_PREFIX == cfg["default"]["deploy"]["parameters"]["s3_prefix"]
