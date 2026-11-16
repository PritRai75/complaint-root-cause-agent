"""Check the deployed API, and make real runs through it.

From the repo root, on the laptop, after sam deploy (reads the stack's outputs and
the API key with your aws login; the key's value is never printed):
  python -m deploy.smoke                     free checks: no key -> 403, bad body -> 400,
                                             unknown run -> 404 (the agent doesn't run)
  python -m deploy.smoke --run "question"    one real run: submit, poll every 15 s until it
                                             finishes, save the JSON and the report

A real run is saved to scratch/deploy/<run_id>.json (the status as returned) and
scratch/deploy/<run_id>.docx (the report, when there is one).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

STACK = "complaint-agent"
REGION = "ap-south-1"
FINAL_WAIT = 45 * 60
POLL = 15
OUT = os.path.join("scratch", "deploy")


def stack_outputs(cfn):
    stack = cfn.describe_stacks(StackName=STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def call(method, url, key=None, body=None, timeout=35):
    """(status code, parsed JSON or text). Never raises on HTTP errors."""
    data = None if body is None else (body if isinstance(body, bytes) else
                                      json.dumps(body).encode("utf-8"))
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("x-api-key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    try:
        return code, json.loads(raw or b"null")
    except ValueError:
        return code, raw.decode("utf-8", "replace")


def free_checks(api, key, out=print):
    checks = [
        ("POST /runs without a key", call("POST", f"{api}/runs", None, {"question": "x"}), 403),
        ("POST /runs with an empty body", call("POST", f"{api}/runs", key, b"{}"), 400),
        ("POST /runs with a 1,001-character question",
         call("POST", f"{api}/runs", key, {"question": "a" * 1001}), 400),
        ("GET /runs/<unknown run>", call("GET", f"{api}/runs/run-{'0' * 32}", key), 404),
        ("GET /runs/<badly formed ID>", call("GET", f"{api}/runs/nope", key), 400),
    ]
    failed = 0
    for name, (code, _), want in checks:
        ok = code == want
        failed += not ok
        out(f"  {'ok  ' if ok else 'FAIL'} {name}: {code} (expected {want})")
    return failed


def real_run(api, key, question, out=print, sleep=time.sleep, clock=time.monotonic):
    code, body = call("POST", f"{api}/runs", key, {"question": question})
    out(f"POST /runs: {code} {json.dumps(body)}")
    if code != 202:
        return 1
    run_id = body["run_id"]
    started = clock()
    while True:
        sleep(POLL)
        code, st = call("GET", f"{api}/runs/{run_id}", key)
        elapsed = clock() - started
        status = st.get("status") if isinstance(st, dict) else None
        out(f"  {elapsed:6.0f} s  {code} {status}")
        if code == 200 and status not in ("queued", "running"):
            break
        if elapsed > FINAL_WAIT:
            out("Gave up waiting (45 minutes).")
            return 1
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    url = st.pop("report_url", None)
    out(json.dumps({k: v for k, v in st.items()}, indent=1, ensure_ascii=False))
    if url:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        path = os.path.join(OUT, f"{run_id}.docx")
        with open(path, "wb") as f:
            f.write(data)
        out(f"Report saved to {path} ({len(data):,} bytes)")
    out(f"Status saved to {os.path.join(OUT, run_id + '.json')}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the deployed API; --run makes one real run.")
    ap.add_argument("--run", metavar="QUESTION", help="submit one real question (costs money)")
    args = ap.parse_args(argv)
    import boto3
    outputs = stack_outputs(boto3.client("cloudformation", region_name=REGION))
    api = outputs["ApiUrl"].rstrip("/")
    key = boto3.client("apigateway", region_name=REGION).get_api_key(
        apiKey=outputs["ApiKeyId"], includeValue=True)["value"]
    print(f"API {api} (key: {len(key)} characters)")
    if args.run:
        return real_run(api, key, args.run)
    failed = free_checks(api, key)
    print("All free checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
