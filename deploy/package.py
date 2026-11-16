"""Build and upload the agent's zip for AgentCore Runtime (direct code deployment).

From the repo root, on the laptop (needs the built databases in data/):
  python -m deploy.package build     build/agent.zip, checked; no AWS calls
  python -m deploy.package upload    copy it to SAM's bucket; writes build/code-bucket.txt
                                     and build/code-key.txt for sam deploy

What goes in the zip (decision, S10: the databases ship inside it; the data is
frozen, so they change only with a rebuild):
  the dependencies   requirements.txt installed for Linux ARM64, Python 3.13, as
                     wheels only (no builds). numpy, pandas and matplotlib at our
                     pins publish ARM64 wheels for glibc 2.28 or newer only, so the
                     manylinux_2_28 tag is included (the runtime is Amazon Linux
                     2023, glibc 2.34); the AWS example's manylinux2014 alone
                     would refuse them.
  the code           agent/, ingest/*.py, eval/taxonomy.py and its two data files,
                     deploy/agent_app.py, and main.py (starts the server)
  the data           data/agent.sqlite and data/agent_text.sqlite, checked
                     against agent/semantic.toml; data/tag_cache.sqlite if present
                     (tag codes and token counts keyed by hashes; no text)

Checks: no AWS CRT in the requirements (the runtime uses its role); every
compiled library is ARM64; at most 750 MB unzipped and 250 MB zipped (AgentCore's
limits for a zip); permissions 644 for files and 755 for folders (AgentCore
needs them readable). The zip is deterministic: sorted entries and a fixed date,
so the same inputs give the same SHA-256, and the upload key carries it.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
STAGE = BUILD / "agent"
ZIP = BUILD / "agent.zip"
MAX_UNZIPPED = 750 * 1024 * 1024
MAX_ZIPPED = 250 * 1024 * 1024
PLATFORMS = ["manylinux2014_aarch64"] + [f"manylinux_2_{m}_aarch64" for m in range(17, 29)]
PYTHON = "3.13"
CODE_GLOBS = ["agent/*.py", "agent/semantic.toml", "ingest/*.py", "deploy/__init__.py",
              "deploy/agent_app.py"]
CODE_FILES = ["eval/__init__.py", "eval/taxonomy.py", "eval/taxonomy.toml",
              "eval/taxonomy_sample.json"]
DATA_FILES = ["data/agent.sqlite", "data/agent_text.sqlite"]
OPTIONAL_DATA = ["data/tag_cache.sqlite"]
MAIN_PY = ('"""Entry point for AgentCore Runtime: python main.py (written by deploy/package.py)."""\n'
           "from deploy.agent_app import main\n\nmain()\n")
CRT = re.compile(r"awscrt|\[\s*crt\s*\]", re.IGNORECASE)
EM_AARCH64 = 183
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
SAM_BUCKET_STACK = "aws-sam-cli-managed-default"
KEY_PREFIX = "complaint-agent"


class PackageError(Exception):
    pass


def check_requirements(text):
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and CRT.search(line):
            raise PackageError(f"requirements.txt asks for the AWS CRT ({line!r}); "
                               "the runtime must not ship it")


def code_files(root=ROOT):
    files = set(CODE_FILES)
    for pattern in CODE_GLOBS:
        files.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    missing = sorted(f for f in files if not (root / f).is_file())
    if missing:
        raise PackageError(f"missing: {', '.join(missing)}")
    return sorted(files)


def elf_machine(path):
    """The ELF machine number of a compiled library, or None if not ELF."""
    with open(path, "rb") as f:
        head = f.read(20)
    if head[:4] != b"\x7fELF":
        return None
    return int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")


def check_binaries(stage):
    wrong = []
    for p in sorted(stage.rglob("*")):
        if p.is_file() and (p.suffix == ".so" or ".so." in p.name):
            m = elf_machine(p)
            if m is not None and m != EM_AARCH64:
                wrong.append(p.relative_to(stage).as_posix())
    if wrong:
        raise PackageError(f"{len(wrong)} compiled files aren't ARM64, e.g. {wrong[0]}")


def check_databases(root=ROOT):
    """The shipped databases must be the ones built from this layer, and the
    tag cache must hold no text column."""
    sys.path.insert(0, str(root))
    from agent import semantic as sem
    cwd = os.getcwd()
    os.chdir(root)
    try:
        conn, _ = sem.connect()
        conn.close()
        sem.connect_text().close()
    except sem.LayerError as e:
        raise PackageError(str(e)) from e
    finally:
        os.chdir(cwd)
    cache = root / "data" / "tag_cache.sqlite"
    if cache.exists():
        conn = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(tags)")}
        finally:
            conn.close()
        if cols & {"narrative", "text", "user", "message"}:
            raise PackageError(f"data/tag_cache.sqlite has a text column ({sorted(cols)})")


def pip_install(requirements, target):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
           "--target", str(target), "--implementation", "cp", "--python-version", PYTHON,
           "--only-binary=:all:", "--no-compile", "-r", str(requirements)]
    for plat in PLATFORMS:
        cmd += ["--platform", plat]
    subprocess.run(cmd, check=True)


def tree_size(stage):
    return sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())


def write_zip(stage, path):
    """Deterministic zip: sorted names, fixed date, 644 files and 755 folders."""
    entries = sorted(p for p in stage.rglob("*"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in entries:
            rel = p.relative_to(stage).as_posix()
            if p.is_dir():
                info = zipfile.ZipInfo(rel + "/", FIXED_DATE)
                info.external_attr = (0o40755 << 16) | 0x10
                z.writestr(info, b"")
            else:
                info = zipfile.ZipInfo(rel, FIXED_DATE)
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(p, "rb") as f:
                    z.writestr(info, f.read(), compresslevel=6)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(root=ROOT, install=pip_install, check_data=check_databases,
          max_unzipped=MAX_UNZIPPED, max_zipped=MAX_ZIPPED, out=print):
    root = Path(root)
    build_dir, stage, zpath = root / "build", root / "build" / "agent", root / "build" / "agent.zip"
    requirements = root / "requirements.txt"
    check_requirements(requirements.read_text(encoding="utf-8"))
    code = code_files(root)
    missing = [f for f in DATA_FILES if not (root / f).is_file()]
    if missing:
        raise PackageError(f"missing: {', '.join(missing)}. Build them with "
                           "python -m agent.semantic build")
    check_data(root)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    out(f"Installing requirements.txt for Linux ARM64, Python {PYTHON} (wheels only)...")
    install(requirements, stage)
    check_binaries(stage)
    shipped = []
    for rel in code + DATA_FILES + [f for f in OPTIONAL_DATA if (root / f).is_file()]:
        dest = stage / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dest)
        shipped.append(rel)
    (stage / "main.py").write_text(MAIN_PY, encoding="utf-8", newline="\n")
    for cache in list(stage.rglob("__pycache__")):
        shutil.rmtree(cache)
    unzipped = tree_size(stage)
    if unzipped > max_unzipped:
        raise PackageError(f"unzipped size {unzipped / 2**20:.1f} MB is over "
                           f"{max_unzipped / 2**20:.0f} MB")
    if zpath.exists():
        zpath.unlink()
    write_zip(stage, zpath)
    zipped = zpath.stat().st_size
    if zipped > max_zipped:
        raise PackageError(f"zip size {zipped / 2**20:.1f} MB is over {max_zipped / 2**20:.0f} MB")
    digest = sha256_file(zpath)
    manifest = {"zip_sha256": digest, "zip_bytes": zipped, "unzipped_bytes": unzipped,
                "files": {rel: sha256_file(root / rel) for rel in shipped},
                "python": PYTHON, "platforms": PLATFORMS, "entry_point": "main.py"}
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    (build_dir / "agent.zip.sha256").write_text(f"{digest}  agent.zip\n", encoding="utf-8",
                                                newline="\n")
    out(f"build/agent.zip: {zipped / 2**20:.1f} MB zipped (limit 250), "
        f"{unzipped / 2**20:.1f} MB unzipped (limit 750)")
    for rel in DATA_FILES + OPTIONAL_DATA:
        if rel in shipped:
            out(f"  {rel:<28} {(root / rel).stat().st_size / 2**20:7.1f} MB  "
                f"{manifest['files'][rel][:12]}")
    out(f"SHA-256 {digest}")
    return zpath, manifest


def sam_bucket(cfn):
    stacks = cfn.describe_stacks(StackName=SAM_BUCKET_STACK)["Stacks"]
    for o in stacks[0].get("Outputs", []):
        if o["OutputKey"] == "SourceBucket":
            return o["OutputValue"]
    raise PackageError(f"stack {SAM_BUCKET_STACK} has no SourceBucket output")


def upload(root=ROOT, cfn=None, s3=None, out=print):
    root = Path(root)
    zpath = root / "build" / "agent.zip"
    if not zpath.exists():
        raise PackageError("build/agent.zip not found: run python -m deploy.package build")
    digest = sha256_file(zpath)
    if cfn is None or s3 is None:
        import boto3
        cfn = cfn or boto3.client("cloudformation", region_name="ap-south-1")
        s3 = s3 or boto3.client("s3", region_name="ap-south-1")
    bucket = sam_bucket(cfn)
    key = f"{KEY_PREFIX}/agent-{digest[:16]}.zip"
    try:
        s3.head_object(Bucket=bucket, Key=key)
        out(f"Already uploaded: {key}")
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise
        out(f"Uploading {zpath.stat().st_size / 2**20:.1f} MB to SAM's bucket as {key} ...")
        s3.upload_file(str(zpath), bucket, key)
    (root / "build" / "code-bucket.txt").write_text(bucket, encoding="utf-8", newline="")
    (root / "build" / "code-key.txt").write_text(key, encoding="utf-8", newline="")
    out("Wrote build/code-bucket.txt and build/code-key.txt (the bucket name holds your "
        "account ID: don't paste it).")
    return bucket, key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["build", "upload"])
    args = ap.parse_args(argv)
    try:
        if args.action == "build":
            build()
        else:
            upload()
    except (PackageError, subprocess.CalledProcessError) as e:
        print(f"ERROR {e}")
        return 1
    except Exception as e:
        print(f"ERROR {type(e).__name__}: "
              f"{getattr(e, 'response', {}).get('Error', {}).get('Code', '')}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
