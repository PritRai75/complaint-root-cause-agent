"""The API's two Lambda functions (session 10). API Gateway checks the API key,
the throttle and the daily request quota before either runs.

POST /runs       submit: {"question": "..."} -> 202 {"run_id": ..., "status": "queued"}
GET /runs/{id}   status: the run's status; when it has finished, its title,
                 findings or clarifying question, and a link to the .docx
                 (valid for an hour)

submit checks, in order: the body (JSON, a question of 1 to 1,000 characters);
the daily run cap (DAILY_RUNS runs started per UTC day, counted from markers in
the bucket); one run at a time (a queued or running run younger than
STALE_MINUTES blocks a new one: two root-cause runs at once would ask Bedrock for
about 80 tag calls a minute against a quota of 50). Then it writes the run's
status, starts the agent through InvokeAgentRuntime with the run ID as the
runtime session ID, and returns. The agent replies at once and keeps working.

The caps count runs, not requests: polling GET /runs/{id} doesn't use them up.
Two submits in the same instant could both pass the checks (no lock); the API's
throttle (1 a second) makes that unlikely, and the cost of it is one extra run.

Logs hold IDs, statuses and AWS error codes only, never the question.
"""
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config

RUN_ID = re.compile(r"^run-[0-9a-f]{32}$")
MAX_QUESTION = 1_000
MAX_BODY = 8_000
STALE_MINUTES = 45
LINK_SECONDS = 3_600
ACTIVE = ("queued", "running")
FINAL_FIELDS = ("question", "clarifying_question", "final_text", "stop_reason", "counts",
                "seconds", "error")


class Clients:
    """boto3 clients, created on first use (tests pass fakes)."""

    def __init__(self, s3=None, agentcore=None):
        self._s3, self._agentcore = s3, agentcore

    @property
    def s3(self):
        if self._s3 is None:
            region = os.environ.get("AWS_REGION", "ap-south-1")
            self._s3 = boto3.client("s3", region_name=region, config=Config(
                signature_version="s3v4", s3={"addressing_style": "virtual"}))
        return self._s3

    @property
    def agentcore(self):
        if self._agentcore is None:
            self._agentcore = boto3.client("bedrock-agentcore", config=Config(
                read_timeout=60, retries={"max_attempts": 2, "mode": "standard"}))
        return self._agentcore


CLIENTS = Clients()


def now():
    return datetime.now(timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def reply(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body)}


def log(**fields):
    print(json.dumps(fields, sort_keys=True))


def error_code(exc):
    try:
        return exc.response["Error"]["Code"]
    except (AttributeError, KeyError, TypeError):
        return type(exc).__name__


def get_json(s3, bucket, key):
    """The object as JSON, or None if it doesn't exist."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if error_code(exc) in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return json.loads(obj["Body"].read())


def put_json(s3, bucket, key, obj):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj).encode("utf-8"),
                  ContentType="application/json")


def runs_on(s3, bucket, day):
    """Run IDs started on a UTC day (YYYY-MM-DD), from the day markers."""
    out = s3.list_objects_v2(Bucket=bucket, Prefix=f"days/{day}/", MaxKeys=1000)
    return [c["Key"].rsplit("/", 1)[1] for c in out.get("Contents", [])]


def parse_question(event):
    body = event.get("body")
    if body is None:
        return None, "Send a JSON body like {\"question\": \"...\"}."
    if event.get("isBase64Encoded"):
        return None, "Send the body as plain JSON."
    if len(body) > MAX_BODY:
        return None, f"The body is over {MAX_BODY:,} characters."
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return None, "The body isn't valid JSON."
    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not question.strip():
        return None, "\"question\" must be a non-empty string."
    question = " ".join(question.split())
    if len(question) > MAX_QUESTION:
        return None, f"\"question\" is over {MAX_QUESTION:,} characters."
    return question, None


def submit(event, context, clients=CLIENTS, environ=None, clock=now, new_id=None):
    env = environ if environ is not None else os.environ
    bucket, arn = env["RESULTS_BUCKET"], env["AGENT_RUNTIME_ARN"]
    cap = int(env.get("DAILY_RUNS", "3"))
    question, problem = parse_question(event)
    if problem:
        return reply(400, {"error": problem})
    s3, t = clients.s3, clock()
    today, yesterday = t.strftime("%Y-%m-%d"), (t - timedelta(days=1)).strftime("%Y-%m-%d")
    started_today = runs_on(s3, bucket, today)
    if len(started_today) >= cap:
        log(event="refused", reason="daily_cap", runs_today=len(started_today))
        return reply(429, {"error": f"The daily limit of {cap} runs is used up. "
                                    "Try again after midnight UTC."})
    for run_id in started_today + runs_on(s3, bucket, yesterday):
        st = get_json(s3, bucket, f"runs/{run_id}/status.json")
        if not st or st.get("status") not in ACTIVE:
            continue
        since = st.get("started_at") or st.get("queued_at")
        try:
            age = t - datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if age < timedelta(minutes=STALE_MINUTES):
            log(event="refused", reason="busy", active=run_id)
            return reply(429, {"error": "Another run is in progress. Try again in a few "
                                        "minutes.", "active_run": run_id})

    run_id = new_id() if new_id else "run-" + uuid.uuid4().hex
    put_json(s3, bucket, f"runs/{run_id}/status.json",
             {"run_id": run_id, "status": "queued", "queued_at": iso(t)})
    s3.put_object(Bucket=bucket, Key=f"days/{today}/{run_id}", Body=b"")
    try:
        resp = clients.agentcore.invoke_agent_runtime(
            agentRuntimeArn=arn, runtimeSessionId=run_id, qualifier="DEFAULT",
            contentType="application/json", accept="application/json",
            payload=json.dumps({"run_id": run_id, "question": question}).encode("utf-8"))
        answer = json.loads(resp["response"].read() or b"{}")
        if answer.get("status") != "started":
            raise RuntimeError(f"agent answered {answer.get('status')!r}")
    except Exception as exc:
        code = error_code(exc)
        put_json(s3, bucket, f"runs/{run_id}/status.json",
                 {"run_id": run_id, "status": "error", "error": code, "queued_at": iso(t),
                  "finished_at": iso(clock())})
        log(event="start_failed", run_id=run_id, error=code)
        return reply(502, {"error": "The agent could not be started.", "code": code,
                           "run_id": run_id})
    log(event="queued", run_id=run_id, runs_today=len(started_today) + 1)
    return reply(202, {"run_id": run_id, "status": "queued", "poll": f"/runs/{run_id}"})


def status(event, context, clients=CLIENTS, environ=None):
    env = environ if environ is not None else os.environ
    bucket = env["RESULTS_BUCKET"]
    run_id = (event.get("pathParameters") or {}).get("run_id", "")
    if not RUN_ID.match(run_id or ""):
        return reply(400, {"error": "A run ID looks like run-<32 lowercase hex>."})
    s3 = clients.s3
    st = get_json(s3, bucket, f"runs/{run_id}/status.json")
    if st is None:
        return reply(404, {"error": "No run with that ID."})
    body = dict(st)
    if st.get("status") in ACTIVE:
        return reply(200, body)
    result = get_json(s3, bucket, f"runs/{run_id}/result.json") or {}
    for k in FINAL_FIELDS:
        if k in result and k not in body:
            body[k] = result[k]
    sent = result.get("sent")
    if isinstance(sent, dict):           # sender kind, message ID or AWS error code; no addresses
        body["sent"] = {k: sent[k] for k in ("sender", "message_id", "error") if k in sent}
    report = result.get("report")
    if report:
        body["report"] = {k: report.get(k) for k in ("title", "findings", "filename", "bytes",
                                                    "sha256")}
    if st.get("has_report") and report:
        body["report_url"] = s3.generate_presigned_url(
            "get_object", ExpiresIn=LINK_SECONDS,
            Params={"Bucket": bucket, "Key": f"runs/{run_id}/report.docx",
                    "ResponseContentDisposition":
                        f"attachment; filename=\"{report.get('filename', 'report.docx')}\""})
        body["report_url_expires_in"] = LINK_SECONDS
    return reply(200, body)
