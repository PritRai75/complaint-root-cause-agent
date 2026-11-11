"""send_report: email the run's Word report to one fixed address, once per run.

What code guarantees (and the model can't change):
  recipient  the sender object holds it, set from configuration (REPORT_TO) when
             the graph is built; the tool takes no address from the model
  once       at most one send attempt per run, counted before the attempt, so a
             failure that may have reached SES is never retried
  content    the subject, body and attachment are built by code from the
             report in the store (the checked title and findings)

DryRunSender records what would have been sent (recipient, subject,
attachment name, size and SHA-256) for the eval to grade "exactly one email";
it never touches the network. SesSender sends through Amazon SES v2 in
ap-south-1 with the role's or profile's credentials (t6-ses: one real send for
the demo). Failures are reported with the AWS error code only.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from email.message import EmailMessage

from agent import report as rp
from agent import tagger as tg

REGION = "ap-south-1"
TO_ENV, FROM_ENV = "REPORT_TO", "REPORT_FROM"
ADDRESS = re.compile(r"^[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[A-Za-z]{2,}$")
MAX_SUBJECT = 150
DOCX_TYPE = ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document")


class SendError(Exception):
    pass


def address(name, environ=None):
    """One email address from the environment variable `name`, or SendError."""
    value = (environ if environ is not None else os.environ).get(name, "").strip()
    if not ADDRESS.match(value):
        raise SendError(f"{name} must be set to one email address")
    return value


def summary(message):
    """What was (or would be) sent, without the attachment's content."""
    files = []
    for part in message.iter_attachments():
        data = part.get_content()
        files.append({"filename": part.get_filename(), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return {"to": message["To"], "from": message["From"], "subject": message["Subject"],
            "attachments": files,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


class DryRunSender:
    """Records sends instead of making them. log_path: optional JSON-lines file."""

    kind = "dry_run"

    def __init__(self, to="recipient@example.invalid", sender="agent@example.invalid",
                 log_path=None):
        self.to, self.sender, self.log_path = to, sender, log_path
        self.sent = []

    def send(self, message):
        record = summary(message)
        self.sent.append(record)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return f"dry-run-{len(self.sent)}"


class SesSender:
    """Amazon SES v2. Addresses come from REPORT_TO and REPORT_FROM unless given."""

    kind = "ses"

    def __init__(self, to=None, sender=None, client=None, environ=None):
        self.to = to or address(TO_ENV, environ)
        self.sender = sender or address(FROM_ENV, environ)
        for value in (self.to, self.sender):
            if not ADDRESS.match(value):
                raise SendError("each address must be one email address")
        self.client = client

    def send(self, message):
        if self.client is None:
            import boto3
            self.client = boto3.client("sesv2", region_name=REGION)
        response = self.client.send_email(
            FromEmailAddress=self.sender,
            Destination={"ToAddresses": [self.to]},
            Content={"Raw": {"Data": message.as_bytes()}})
        return response.get("MessageId")


def one_line(text):
    return " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())


def message_for(store, sender):
    r = store.report
    msg = EmailMessage()
    msg["From"] = sender.sender
    msg["To"] = sender.to
    msg["Subject"] = one_line("Complaint report: " + r["title"])[:MAX_SUBJECT]
    lines = [r["title"], ""]
    if store.question:
        lines += ["Question: " + one_line(store.question), ""]
    lines += ["Key findings:"] + [f"- {f}" for f in r["findings"]]
    lines += ["", rp.CAVEAT, "", f"The full report is attached ({r['filename']}).", "",
              f"Sent by {rp.AGENT}."]
    msg.set_content("\n".join(lines))
    msg.add_attachment(r["bytes"], maintype=DOCX_TYPE[0], subtype=DOCX_TYPE[1],
                       filename=r["filename"])
    return msg


def send_report(store, sender):
    """The planner's send tool. Takes nothing from the model: the report is the
    one in the store and the recipient is the sender's. One attempt per run."""
    if store.report is None:
        return rp._err("no_report", "There is no report to send. Call build_report first.")
    if store.send_attempts:
        return rp._err("already_sent", "The report was already sent in this run. "
                                       "One email per run.")
    store.send_attempts += 1
    message = message_for(store, sender)
    record = {"sender": sender.kind, "subject": message["Subject"],
              "attachment_sha256": store.report["sha256"]}
    try:
        record["message_id"] = sender.send(message)
    except Exception as exc:           # the code only: messages can echo the request
        record["error"] = tg.error_code(exc)
        store.sent = record
        return rp._err("send_failed", f"Sending failed ({record['error']}). Not retried: "
                                      "one send attempt per run.")
    store.sent = record
    return {"ok": True, "sender": sender.kind,
            "text": f"Report emailed to the fixed recipient ({sender.kind}). The run is done."}
