"""The agent's HTTP server on AgentCore Runtime (session 10).

Decisions (Prit, S10): direct code deployment (a zip with the code, its ARM64
dependencies and the two databases); the run happens as a background task, so
/invocations answers at once and the run can take longer than any request
limit; the endpoints are written in FastAPI (no AgentCore SDK).

The service contract (AgentCore Runtime, HTTP protocol): listen on 0.0.0.0:8080;
POST /invocations takes the request; GET /ping answers {"status": "Healthy" or
"HealthyBusy", "time_of_last_update": unix seconds}. While the status is
HealthyBusy the session is kept alive. time_of_last_update changes only when the
status changes (the docs warn against stamping every ping).

One run per session: the submit function uses the run ID as the runtime
session ID, so every run gets its own microVM. /invocations takes
{"run_id": "run-<32 hex>", "question": "..."}, starts the run on a thread and
returns {"status": "started"}. The run writes to S3 (RESULTS_BUCKET):
    runs/<run_id>/status.json   running, then the final status
    runs/<run_id>/result.json   what the agent did (no narrative text)
    runs/<run_id>/report.docx   the report, when one was built
and code emails the report once to the fixed recipient (REPORT_TO, through SES).

Logs (stdout, collected by AgentCore) hold numbers and codes only: no question,
no findings, no SQL, never AWS error messages (decision carried over from
Project 1). The question and results live in the private bucket.

Started by the package's main.py: python main.py
"""
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, Header
from fastapi.responses import JSONResponse

HOST, PORT = "0.0.0.0", 8080
RUN_ID = re.compile(r"^run-[0-9a-f]{32}$")
MAX_QUESTION = 1_000                 # the graph's own limit (tests check they match)
BUCKET_ENV = "RESULTS_BUCKET"
PREFIX = "runs"
TMP = "/tmp/complaint-agent"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(**fields):
    """One JSON line on stdout: numbers, codes and IDs only."""
    print(json.dumps({"at": now_iso(), **fields}, sort_keys=True), flush=True)


class Health:
    """What /ping reports. Busy while a run is active."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.active = set()
        self.changed = int(clock())

    def status(self):
        with self.lock:
            return {"status": "HealthyBusy" if self.active else "Healthy",
                    "time_of_last_update": self.changed}

    def start(self, run_id):
        """Mark run_id active. False if a run is already active here."""
        with self.lock:
            if self.active:
                return False
            self.active.add(run_id)
            self.changed = int(self.clock())
            return True

    def end(self, run_id):
        with self.lock:
            self.active.discard(run_id)
            if not self.active:
                self.changed = int(self.clock())


class Results:
    """The run's files in the private results bucket."""

    def __init__(self, bucket=None, client=None):
        self.bucket = bucket or os.environ.get(BUCKET_ENV, "")
        if not self.bucket:
            raise RuntimeError(f"{BUCKET_ENV} is not set")
        self.client = client

    def _s3(self):
        if self.client is None:
            import boto3
            self.client = boto3.client("s3", region_name=os.environ.get("AWS_REGION")
                                       or os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1")
        return self.client

    def put(self, run_id, name, data, content_type):
        self._s3().put_object(Bucket=self.bucket, Key=f"{PREFIX}/{run_id}/{name}",
                              Body=data, ContentType=content_type)

    def put_json(self, run_id, name, obj):
        self.put(run_id, name, json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8"),
                 "application/json")


def public_result(run_id, result):
    """What the API returns about a run: the graph's result without anything
    that could hold narrative text (there is none in it today; quotes are
    stored as code and complaint ID only)."""
    keep = ("status", "stop_reason", "model_error", "question", "clarifying_question",
            "final_text", "report", "counts", "tokens", "queries", "tag_queries", "model",
            "plan_version", "prompt_sha256", "tools_sha256", "seconds", "error")
    out = {"run_id": run_id, **{k: result.get(k) for k in keep if k in result}}
    sent = result.get("sent")
    if isinstance(sent, dict):
        out["sent"] = {k: sent.get(k) for k in ("sender", "message_id", "error") if k in sent}
    return out


def ses_sender():
    from agent import send as sd
    return sd.SesSender()


def graph_answer(question, sender, tag_cache):
    from agent import graph
    return graph.answer(question, sender, tag_cache=tag_cache)


def error_code(exc):
    try:
        return exc.response["Error"]["Code"]
    except (AttributeError, KeyError, TypeError):
        return type(exc).__name__


def run_job(run_id, question, results, health, answer=graph_answer, sender_factory=ses_sender,
            tag_cache=None, clock=time.monotonic):
    """One run, start to finish, on its own thread. Never raises."""
    started = clock()
    run = None
    try:
        results.put_json(run_id, "status.json", {"run_id": run_id, "status": "running",
                                                 "started_at": now_iso()})
        log(event="run_started", run_id=run_id)
        result, run = answer(question, sender_factory(), tag_cache)
        report = run.store.report if run is not None else None
        if report is not None:
            results.put(run_id, "report.docx", report["bytes"], DOCX)
        results.put_json(run_id, "result.json", public_result(run_id, result))
        status = result.get("status", "error")
        counts, tokens = result.get("counts") or {}, result.get("tokens") or {}
        results.put_json(run_id, "status.json", {
            "run_id": run_id, "status": status, "finished_at": now_iso(),
            "seconds": round(clock() - started, 1), "has_report": report is not None})
        log(event="run_finished", run_id=run_id, status=status,
            seconds=round(clock() - started, 1), turns=counts.get("turns"),
            tool_calls=counts.get("tool_calls"), tag_runs=counts.get("tag_runs"),
            failed_calls=counts.get("failed_calls"), input_tokens=tokens.get("input"),
            output_tokens=tokens.get("output"), has_report=report is not None)
    except Exception as exc:                   # the code only: messages can echo the request
        code = error_code(exc)
        log(event="run_failed", run_id=run_id, error=code, seconds=round(clock() - started, 1))
        try:
            results.put_json(run_id, "status.json", {
                "run_id": run_id, "status": "error", "error": code, "finished_at": now_iso(),
                "seconds": round(clock() - started, 1), "has_report": False})
        except Exception as exc2:
            log(event="status_write_failed", run_id=run_id, error=error_code(exc2))
    finally:
        if run is not None:
            try:
                run.close()
            except Exception as exc:
                log(event="close_failed", run_id=run_id, error=error_code(exc))
        health.end(run_id)


def bad(message):
    return JSONResponse({"error": message}, status_code=400)


def create_app(health=None, results_factory=Results, start=None, tag_cache=None):
    """The FastAPI app. start(target, args) runs a run in the background (a
    daemon thread by default; tests pass their own)."""
    health = health or Health()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.health = health

    def start_thread(target, args):
        threading.Thread(target=target, args=args, daemon=True,
                         name=f"run-{args[0][-8:]}").start()
    start = start or start_thread

    @app.get("/ping")
    def ping():
        return health.status()

    @app.post("/invocations")
    def invocations(body: dict = Body(...),
                    session_id: Optional[str] = Header(
                        None, alias="X-Amzn-Bedrock-AgentCore-Runtime-Session-Id")):
        run_id, question = body.get("run_id"), body.get("question")
        if not isinstance(run_id, str) or not RUN_ID.match(run_id):
            return bad("run_id must look like run-<32 lowercase hex>")
        if session_id is not None and session_id != run_id:
            return bad("the runtime session ID must be the run ID")
        if not isinstance(question, str) or not question.strip():
            return bad("question must be a non-empty string")
        if len(question) > MAX_QUESTION:
            return bad(f"question is over {MAX_QUESTION:,} characters")
        if not health.start(run_id):
            return JSONResponse({"status": "busy", "run_id": run_id}, status_code=409)
        try:
            results = results_factory()
            start(run_job, (run_id, question, results, health, graph_answer, ses_sender,
                            tag_cache))
        except Exception as exc:
            health.end(run_id)
            log(event="start_failed", run_id=run_id, error=error_code(exc))
            return JSONResponse({"status": "error", "error": error_code(exc)}, status_code=500)
        log(event="run_accepted", run_id=run_id)
        return {"status": "started", "run_id": run_id}

    return app


def prepare(root, tmp=TMP):
    """Process start in the runtime: work from the package root (the code finds
    agent/, eval/ and data/ by relative path), keep matplotlib's cache in /tmp,
    and copy the shipped tag cache to /tmp, where the tagger may write to it."""
    os.chdir(root)
    os.makedirs(tmp, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp, "matplotlib"))
    shipped = Path(root) / "data" / "tag_cache.sqlite"
    cache = os.path.join(tmp, "tag_cache.sqlite")
    if shipped.exists():
        shutil.copyfile(shipped, cache)
    return cache


def main():
    import uvicorn
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    cache = prepare(root)
    log(event="server_starting", port=PORT)
    uvicorn.run(create_app(tag_cache=cache), host=HOST, port=PORT, log_level="warning",
                access_log=False)


if __name__ == "__main__":
    main()
