"""Checks on the real build. Skipped when the data is not on this machine (as in CI)."""
import json
import os
import sqlite3

import pytest

from ingest import sources
from ingest.fetch import load_manifest, sha256_of

# Counts read off the archive probe output on 3 Nov 2026, card plus checking,
# January to August 2026. A different build result means something changed.
EXPECTED_COMPLAINTS = 125_220
EXPECTED_NARRATIVES = 41_770

needs_db = pytest.mark.skipif(not os.path.exists(sources.DB_PATH),
                              reason="database not built on this machine")
needs_raw = pytest.mark.skipif(
    not os.path.exists(sources.MANIFEST)
    or not all(os.path.exists(os.path.join(sources.RAW_DIR, f)) for f in sources.FILES),
    reason="raw archive files or manifest not on this machine")


def test_manifest_lists_every_source():
    if not os.path.exists(sources.MANIFEST):
        pytest.skip("manifest not created yet")
    manifest = load_manifest(sources.MANIFEST)
    assert sorted(manifest["files"]) == sorted(sources.FILES)


@needs_raw
def test_raw_files_match_manifest():
    manifest = load_manifest(sources.MANIFEST)
    for name in sources.FILES:
        path = os.path.join(sources.RAW_DIR, name)
        assert os.path.getsize(path) == manifest["files"][name]["bytes"], name
        assert sha256_of(path) == manifest["files"][name]["sha256"], name


@pytest.fixture
def conn():
    c = sqlite3.connect(f"file:{sources.DB_PATH}?mode=ro", uri=True)
    yield c
    c.close()


@needs_db
def test_counts(conn):
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    n_c = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    n_n = conn.execute("SELECT COUNT(*) FROM narratives").fetchone()[0]
    assert (n_c, n_n) == (int(info["complaints_rows"]), int(info["narratives_rows"]))
    assert (n_c, n_n) == (EXPECTED_COMPLAINTS, EXPECTED_NARRATIVES)


@needs_db
def test_scope(conn):
    products = {r[0] for r in conn.execute("SELECT DISTINCT product FROM complaints")}
    assert products == set(sources.PRODUCTS)
    lo, hi = conn.execute("SELECT MIN(date_received), MAX(date_received) FROM complaints").fetchone()
    assert sources.DATE_FROM <= lo and hi <= sources.DATE_TO
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    used = [s["file"] for s in json.loads(info["source_files"])]
    assert used == sources.FILES


@needs_db
def test_tables_agree(conn):
    flagged = conn.execute("SELECT COUNT(*) FROM complaints WHERE has_narrative = 1").fetchone()[0]
    orphans = conn.execute(
        "SELECT COUNT(*) FROM narratives n LEFT JOIN complaints c USING (complaint_id) "
        "WHERE c.complaint_id IS NULL").fetchone()[0]
    n_n = conn.execute("SELECT COUNT(*) FROM narratives").fetchone()[0]
    assert flagged == n_n and orphans == 0
