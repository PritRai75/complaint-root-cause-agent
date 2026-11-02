"""Download the archive zips we use and record them in a manifest.

Usage: python -m ingest.fetch

For each file in sources.FILES
  - if it is missing from data/raw, download it
  - work out its size and SHA-256
  - if the manifest already has it, stop with an error when they differ
  - otherwise add it to the manifest
Re-running is safe. A file that has changed is never accepted quietly.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from ingest import sources

RETRY_WAITS = (5, 10)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_download(url, dest):
    """Save url to dest exactly as served. Retries on 429 and 5xx."""
    tmp = dest + ".part"
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "complaint-agent-fetch"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            os.replace(tmp, dest)
            return
        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or e.code >= 500
            if not retryable or attempt == len(RETRY_WAITS):
                raise
            wait = RETRY_WAITS[attempt]
            print(f"  HTTP {e.code}, retrying in {wait} s")
            time.sleep(wait)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def load_manifest(path):
    if not os.path.exists(path):
        return {"archive_page": sources.ARCHIVE_PAGE, "files": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def fetch(files=None, raw_dir=sources.RAW_DIR, manifest_path=sources.MANIFEST,
          download=http_download):
    files = files if files is not None else sources.FILES
    os.makedirs(raw_dir, exist_ok=True)
    manifest = load_manifest(manifest_path)
    problems = []
    for name in files:
        path = os.path.join(raw_dir, name)
        url = sources.url_for(name)
        how = "local file"
        if not os.path.exists(path):
            print(f"Downloading {name}")
            download(url, path)
            how = "downloaded"
        size = os.path.getsize(path)
        digest = sha256_of(path)
        known = manifest["files"].get(name)
        if known:
            if known["sha256"] != digest or known["bytes"] != size:
                problems.append(f"{name}: differs from manifest "
                                f"(manifest {known['bytes']:,} bytes {known['sha256'][:12]}, "
                                f"local {size:,} bytes {digest[:12]})")
                continue
            print(f"OK        {name}  {size:,} bytes")
        else:
            manifest["files"][name] = {
                "url": url,
                "bytes": size,
                "sha256": digest,
                "recorded_at": now_utc(),
                "recorded_from": how,
            }
            print(f"Recorded  {name}  {size:,} bytes")
    save_manifest(manifest, manifest_path)
    if problems:
        for p in problems:
            print("ERROR", p)
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("usage: python -m ingest.fetch")
    fetch()
