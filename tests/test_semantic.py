"""Semantic layer: the layer file, the agent database build, the guard and the metrics."""
import os
import shutil
import sqlite3

import pytest

from agent import semantic as sem
from ingest import build_db, sources

CARD, CHK = sources.PRODUCTS
MONTHS = ["2026-0%d" % m for m in range(1, 9)]
RESPONSES = ["Closed with explanation", "Closed with monetary relief",
             "Closed with non-monetary relief", "In progress", "Untimely response"]

# Each layer version's file is pinned, so an edit without a version bump fails
# here, and a database built from the old file is refused by connect().
LAYER_SHA256 = {
    "layer-1": "1d51d469c6691eab011c0aaedfeff40361bf0cca3fbd82153f78fb6090d65d01",
}


def make_source(path):
    """Ten complaints per product and month. Responses cycle through RESPONSES,
    timely is No for untimely ones and for every seventh complaint, and every
    third complaint before August has a narrative."""
    conn = sqlite3.connect(path)
    conn.executescript(build_db.SCHEMA)
    cid = 1000
    for product in sources.PRODUCTS:
        for month in MONTHS:
            for i in range(10):
                cid += 1
                response = RESPONSES[cid % len(RESPONSES)]
                timely = "No" if response == "Untimely response" or cid % 7 == 0 else "Yes"
                has_text = int(cid % 3 == 0 and month != "2026-08")
                conn.execute(
                    f"INSERT INTO complaints ({', '.join(build_db.DB_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(build_db.DB_COLUMNS))})",
                    (cid, f"{month}-1{i % 9}", product, "Sub", f"Issue {i % 2}", None,
                     f"Bank {i % 3}", "TX", None, "Web", f"{month}-1{i % 9}",
                     response, None, timely, has_text, "A.zip"))
                if has_text:
                    text = f"Narrative {cid}"
                    conn.execute("INSERT INTO narratives VALUES (?, ?, ?)", (cid, text, len(text)))
    conn.executemany("INSERT INTO build_info VALUES (?, ?)", [
        ("built_at", "2026-11-02T00:00:00Z"), ("loader_commit", "abc1234")])
    conn.commit()
    conn.close()


@pytest.fixture
def env(tmp_path):
    """Source database, a copy of the layer and the agent database, in a folder
    whose name has a space (like C:\\Users\\Prit Raithatha)."""
    root = tmp_path / "with space"
    root.mkdir()
    src = str(root / "complaints.sqlite")
    make_source(src)
    layer_path = str(root / "semantic.toml")
    shutil.copy(sem.LAYER_PATH, layer_path)
    agent = str(root / "agent.sqlite")
    sem.build(source_path=src, agent_path=agent, layer_path=layer_path)
    return {"src": src, "agent": agent, "layer": layer_path, "root": root}


def source_count(env, where):
    conn = sqlite3.connect(env["src"])
    n = conn.execute(f"SELECT COUNT(*) FROM complaints WHERE {where}").fetchone()[0]
    conn.close()
    return n


@pytest.fixture
def guarded(env):
    conn, guard = sem.connect(agent_path=env["agent"], layer_path=env["layer"])
    yield conn, guard
    conn.close()


# ---------- the layer file ----------

def test_layer_file_loads_and_is_pinned():
    layer = sem.load()
    assert list(layer["tables"]) == ["complaints_v", "responses_v", "narratives_v"]
    assert LAYER_SHA256.get(layer["version"]) == layer["sha256"], (
        "agent/semantic.toml changed: bump version and pin the new sha256 here")


def test_windows_match_the_decisions():
    w = sem.load()["window"]
    assert (w["all"]["from"], w["all"]["to"]) == (sources.DATE_FROM, sources.DATE_TO)
    assert w["response_final"]["to"] == "2026-06-30"   # decision 5, session 2
    assert w["narratives"]["to"] == "2026-07-31"


def test_complaints_v_and_narratives_v_carry_no_response_fields():
    tables = sem.load()["tables"]
    for name in ("complaints_v", "narratives_v"):
        cols = {c["name"] for c in tables[name]["column"]}
        assert not cols & {"company_response", "timely_response", "company_public_response"}


@pytest.mark.parametrize("edit, message", [
    (lambda t: t.replace('version = "layer-1"', 'version = ""'), "version is empty"),
    (lambda t: t.replace('to = "2026-06-30"', 'to = "2026-6-30"'), "not YYYY-MM-DD"),
    (lambda t: t.replace('to = "2026-06-30"', 'to = "2025-06-30"'), "from is after to"),
    (lambda t: t.replace('name = "responses_v"', 'name = "responses"'), "ending in _v"),
    (lambda t: t.replace('name = "responses_v"', 'name = "complaints_v"'), "more than once"),
    (lambda t: t.replace('window = "narratives"\n', 'window = "nope"\n'), "is not defined"),
    (lambda t: t.replace('from = "narratives JOIN complaints USING (complaint_id)"',
                         'from = "narratives JOIN complaints ON 1 = 1"'), "may only join"),
    (lambda t: t.replace('from = "complaints"', 'from = "build_info"', 1), "may only join"),
    (lambda t: t.replace('from = "narratives JOIN complaints USING (complaint_id)"',
                         'from = "narratives JOIN narratives USING (complaint_id)"'), "may only join"),
    (lambda t: t.replace('description = "Month received, YYYY-MM."', 'description = ""', 1),
     "has no description"),
    (lambda t: t.replace('table = "responses_v"', 'table = "nowhere_v"', 1), "is not in the layer"),
    (lambda t: t.replace('kind = "rate"', 'kind = "ratio"', 1), "kind must be"),
    (lambda t: t.replace('denominator = "SUM(closed)"', 'denominator = ""', 1),
     "needs numerator and denominator"),
])
def test_malformed_layer_is_refused(tmp_path, edit, message):
    with open(sem.LAYER_PATH, encoding="utf-8") as f:
        text = f.read()
    bad = edit(text)
    assert bad != text, "the edit did not apply"
    path = tmp_path / "semantic.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(sem.LayerError, match=message):
        sem.load(str(path))


# ---------- the build ----------

def test_build_counts_match_the_source(env):
    conn = sqlite3.connect(env["agent"])
    n = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
         for t in ("complaints_v", "responses_v", "narratives_v")}
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    conn.close()
    assert n["complaints_v"] == source_count(env, "1 = 1") == 160
    assert n["responses_v"] == source_count(
        env, "date_received <= '2026-06-30' AND company_response <> 'In progress'")
    assert n["narratives_v"] == source_count(env, "has_narrative = 1")
    assert info["layer_version"] == "layer-1"
    assert info["layer_sha256"] == sem.file_sha256(env["layer"])
    assert info["source_loader_commit"] == "abc1234"
    assert info["rows_responses_v"] == str(n["responses_v"])


def test_build_keeps_only_the_window(env):
    conn = sqlite3.connect(env["agent"])
    assert conn.execute("SELECT MAX(month) FROM responses_v").fetchone()[0] == "2026-06"
    assert conn.execute(
        "SELECT COUNT(*) FROM responses_v WHERE company_response = 'In progress'").fetchone()[0] == 0
    assert conn.execute("SELECT MAX(month) FROM narratives_v").fetchone()[0] == "2026-07"
    assert conn.execute("SELECT MAX(month) FROM complaints_v").fetchone()[0] == "2026-08"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    assert tables == {"complaints_v", "responses_v", "narratives_v", "build_info"}


def test_build_refuses_to_overwrite(env):
    with pytest.raises(sem.LayerError, match="--force"):
        sem.build(source_path=env["src"], agent_path=env["agent"], layer_path=env["layer"])
    sem.build(source_path=env["src"], agent_path=env["agent"], layer_path=env["layer"], force=True)
    assert not os.path.exists(env["agent"] + ".building")


# ---------- the guard ----------

def test_layer_tables_can_be_queried(guarded):
    conn, _ = guarded
    for t in ("complaints_v", "responses_v", "narratives_v"):
        assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0
    assert conn.execute(
        "SELECT COUNT(*) FROM narratives_v n JOIN responses_v r USING (complaint_id)").fetchone()[0] > 0
    assert conn.execute(
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r WHERE i < 3) "
        "SELECT COUNT(*) FROM r").fetchone()[0] == 3
    # COUNT(*) over a CTE is reported as a read of the CTE's name; it must work.
    assert conn.execute(
        "WITH x AS (SELECT product, COUNT(*) AS n FROM complaints_v GROUP BY product) "
        "SELECT COUNT(*) FROM x").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM json_each('[1, 2]')").fetchone()[0] == 2


def test_a_cte_named_like_a_table_still_sees_only_the_window(guarded):
    # The hole in the TEMP-view design: here there are no base tables to reach.
    conn, _ = guarded
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        conn.execute("WITH responses_v AS (SELECT * FROM complaints) "
                     "SELECT MAX(date_received) FROM responses_v")
    assert conn.execute("WITH x AS (SELECT * FROM responses_v) SELECT MAX(month) FROM x"
                        ).fetchone()[0] == "2026-06"


@pytest.mark.parametrize("sql, reason", [
    ("SELECT COUNT(*) FROM complaints", "no such table"),
    ("SELECT name FROM sqlite_master", "prohibited"),
    ("SELECT name FROM sqlite_schema", "prohibited"),
    ("SELECT COUNT(*) FROM build_info", "not authorized"),
    ("WITH x AS (SELECT * FROM build_info) SELECT COUNT(*) FROM x", "prohibited"),
    ("SELECT * FROM pragma_table_info('complaints_v')", "not authorized"),
    ("PRAGMA query_only = OFF", "not authorized"),
    ("PRAGMA table_info(complaints_v)", "not authorized"),
    ("CREATE TEMP TABLE t (x)", "not authorized"),
    ("CREATE TEMP VIEW v AS SELECT 1", "not authorized"),
    ("INSERT INTO complaints_v (complaint_id) VALUES (1)", "not authorized"),
    ("DELETE FROM complaints_v", "not authorized"),
    ("DROP TABLE complaints_v", "not authorized"),
])
def test_guard_refuses(guarded, sql, reason):
    conn, guard = guarded
    with pytest.raises(sqlite3.DatabaseError, match=reason):
        conn.execute(sql)
    if sql.startswith("PRAGMA"):
        assert guard.last_denial == "only SELECT queries are allowed, not PRAGMA"
    if reason != "no such table":
        assert guard.last_denial


def test_guard_refuses_attach(guarded, env):
    conn, guard = guarded
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute("ATTACH DATABASE ? AS s", (sem.uri(env["src"]),))
    assert guard.last_denial == "only SELECT queries are allowed, not ATTACH"


def test_guard_names_the_tables_it_allows(guarded):
    conn, guard = guarded
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("SELECT * FROM build_info")
    assert guard.last_denial == ("build_info is not part of the semantic layer; "
                                 "the tables are complaints_v, narratives_v, responses_v")


def test_connect_refuses_a_database_from_another_layer(env):
    with open(env["layer"], encoding="utf-8") as f:
        text = f.read()
    with open(env["layer"], "w", encoding="utf-8") as f:
        f.write(text.replace("One row per complaint", "One row for each complaint", 1))
    with pytest.raises(sem.LayerError, match="--force"):
        sem.connect(agent_path=env["agent"], layer_path=env["layer"])


def test_connect_without_a_database(tmp_path):
    with pytest.raises(sem.LayerError, match="agent.semantic build"):
        sem.connect(agent_path=str(tmp_path / "none.sqlite"))


# ---------- metrics ----------

def hand(env, where):
    return source_count(env, where)


def test_rates_match_hand_counts(env, guarded):
    conn, _ = guarded
    layer = sem.load(env["layer"])
    final = "date_received <= '2026-06-30' AND company_response <> 'In progress'"
    rows = {r["product"]: r for r in sem.run_metric(conn, layer, "timely_response_rate", ["product"])}
    for p in sources.PRODUCTS:
        num = hand(env, f"{final} AND product = '{p}' AND timely_response = 'Yes'")
        den = hand(env, f"{final} AND product = '{p}'")
        assert (rows[p]["numerator"], rows[p]["denominator"]) == (num, den)
        assert rows[p]["value"] == pytest.approx(100 * num / den)

    relief = sem.run_metric(conn, layer, "relief_rate", filters={"product": CARD})[0]
    assert relief["numerator"] == hand(
        env, f"{final} AND product = '{CARD}' AND company_response IN "
             "('Closed with monetary relief', 'Closed with non-monetary relief')")
    assert relief["denominator"] == hand(
        env, f"{final} AND product = '{CARD}' AND company_response LIKE 'Closed%'")


def test_coverage_leaves_out_august(env, guarded):
    conn, _ = guarded
    layer = sem.load(env["layer"])
    r = sem.run_metric(conn, layer, "narrative_coverage")[0]
    assert r["denominator"] == hand(env, "date_received <= '2026-07-31'")
    assert r["numerator"] == hand(env, "has_narrative = 1")


def test_share_and_count(env, guarded):
    conn, _ = guarded
    layer = sem.load(env["layer"])
    shares = sem.run_metric(conn, layer, "complaint_share", ["product"],
                            filters={"month": ["2026-01", "2026-02"]})
    assert [r["product"] for r in shares] == sorted(sources.PRODUCTS)
    assert sum(r["value"] for r in shares) == pytest.approx(100)
    assert all(r["denominator"] == 40 for r in shares)
    count = sem.run_metric(conn, layer, "complaint_volume", filters={"product": CHK})[0]
    assert (count["value"], count["denominator"]) == (80, None)


@pytest.mark.parametrize("name, by, filters, message", [
    ("nope", (), None, "unknown metric"),
    ("relief_rate", ("state",), None, "no column state"),
    ("complaint_volume", (), {"timely": 1}, "no column timely"),
    ("complaint_share", (), None, "needs at least one by column"),
])
def test_metric_sql_refuses(name, by, filters, message):
    with pytest.raises(sem.LayerError, match=message):
        sem.metric_sql(sem.load(), name, by, filters)


def test_filters_are_parameters_not_text():
    sql, params = sem.metric_sql(sem.load(), "complaint_volume",
                                 filters={"company": "O'Brien Bank"})
    assert "O'Brien" not in sql and params == ["O'Brien Bank"]


def test_describe_shows_every_table_column_and_metric():
    layer = sem.load()
    text = sem.describe(layer)
    for t in layer["table"]:
        assert f"Table {t['name']}" in text
        for c in t["column"]:
            assert f"- {c['name']}: " in text
    for m in layer["metrics"]:
        assert f"- {m}: " in text


# ---------- real data (skip when data/agent.sqlite is not built, as in CI) ----------

# Read off scratch/s4_metric_options.txt (Prit's laptop, 10 Nov 2026), which
# queried data/complaints.sqlite directly without the layer.
REAL = {
    "rows": {"complaints_v": 125_220, "responses_v": 93_519, "narratives_v": 41_770},
    "timely_response_rate": {CHK: (44_462, 44_794), CARD: (47_974, 48_725)},
    "relief_rate": {CHK: (8_235, 44_729), CARD: (12_785, 48_562)},
    "monetary_relief_rate": {CHK: (5_871, 44_729), CARD: (7_762, 48_562)},
    "coverage_by_month": {"2026-01": (7_580, 17_432), "2026-02": (3_962, 13_960),
                          "2026-03": (6_812, 16_147), "2026-04": (7_380, 15_903),
                          "2026-05": (6_079, 15_116), "2026-06": (6_110, 15_124),
                          "2026-07": (3_847, 15_990)},
}

needs_agent_db = pytest.mark.skipif(not os.path.exists(sem.AGENT_DB_PATH),
                                    reason="data/agent.sqlite not built on this machine")


@needs_agent_db
def test_real_layer_matches_the_direct_queries():
    conn, _ = sem.connect()
    layer = sem.load()
    for t, n in REAL["rows"].items():
        assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == n, t
    for name in ("timely_response_rate", "relief_rate", "monetary_relief_rate"):
        got = {r["product"]: (r["numerator"], r["denominator"])
               for r in sem.run_metric(conn, layer, name, ["product"])}
        assert got == REAL[name], name
    got = {r["month"]: (r["numerator"], r["denominator"])
           for r in sem.run_metric(conn, layer, "narrative_coverage", ["month"])}
    assert got == REAL["coverage_by_month"]
    conn.close()
