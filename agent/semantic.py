"""The semantic layer: the agent's own database and the defined metrics.

Usage:
  python -m agent.semantic check            check agent/semantic.toml
  python -m agent.semantic build [--force]  build data/agent.sqlite from data/complaints.sqlite
  python -m agent.semantic show             print the layer as the agent will see it
  python -m agent.semantic metrics          print every metric by product

agent/semantic.toml lists the tables the agent may query. build makes each one
a real table in data/agent.sqlite, so the agent's database holds those tables
and nothing else: no base tables, no July or August response fields. A window
written into a table therefore holds whatever SQL the agent writes, including
SQL that names a CTE after a table.

connect() opens data/agent.sqlite read-only and installs an authorizer that
allows only SELECT, reads, functions and recursive CTEs. ATTACH (which could
open another file, such as data/complaints.sqlite), PRAGMA and every write
are refused, and so are reads of build_info and SQLite's own schema tables,
so describe() is the agent's only view of the schema. connect() also refuses
a database built from a different version of the layer file.

metric_sql() writes the SQL for a defined metric. The eval harness uses it for
reference numbers; the agent sees the same definitions through describe().
"""
import hashlib
import os
import re
import sqlite3
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from ingest import sources
from ingest.build_db import git_commit

LAYER_PATH = "agent/semantic.toml"
AGENT_DB_PATH = "data/agent.sqlite"

BASE_TABLES = ("complaints", "narratives")
KINDS = ("count", "share", "rate")
# The only FROM clauses a table may use: one base table, or the approved join.
APPROVED_FROM = re.compile(
    r"^(complaints|narratives)( JOIN (complaints|narratives) USING \(complaint_id\))?$")
HIDDEN_TABLES = ("build_info",)
# Authorizer action codes, for readable refusals. Result codes such as
# SQLITE_MISUSE share these numbers, so the names are listed, not looked up.
ACTION_NAMES = {getattr(sqlite3, "SQLITE_" + n): n.replace("_", " ") for n in (
    "CREATE_INDEX", "CREATE_TABLE", "CREATE_TEMP_INDEX", "CREATE_TEMP_TABLE",
    "CREATE_TEMP_TRIGGER", "CREATE_TEMP_VIEW", "CREATE_TRIGGER", "CREATE_VIEW", "DELETE",
    "DROP_INDEX", "DROP_TABLE", "DROP_TEMP_INDEX", "DROP_TEMP_TABLE", "DROP_TEMP_TRIGGER",
    "DROP_TEMP_VIEW", "DROP_TRIGGER", "DROP_VIEW", "INSERT", "PRAGMA", "TRANSACTION",
    "UPDATE", "ATTACH", "DETACH", "ALTER_TABLE", "REINDEX", "ANALYZE",
    "CREATE_VTABLE", "DROP_VTABLE", "SAVEPOINT")}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                   sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


class LayerError(Exception):
    pass


# ---------- the layer file ----------

def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def problems(layer):
    """Return a list of problems, empty when the layer is well formed."""
    out = []
    for key in ("version", "written", "data_note"):
        if not str(layer.get(key, "")).strip():
            out.append(f"top-level {key} is empty")

    windows = layer.get("window", {})
    for wname, w in windows.items():
        for k in ("from", "to"):
            if not ISO_DATE.match(str(w.get(k, ""))):
                out.append(f"window {wname}: {k} is not YYYY-MM-DD")
        if str(w.get("from", "")) > str(w.get("to", "")):
            out.append(f"window {wname}: from is after to")
        if not str(w.get("why", "")).strip():
            out.append(f"window {wname}: why is empty")

    tables = layer.get("table", [])
    tnames = [t.get("name", "") for t in tables]
    if not tables:
        out.append("no tables")
    for name in sorted({n for n in tnames if tnames.count(n) > 1}):
        out.append(f"table {name} appears more than once")
    for t in tables:
        name = t.get("name", "")
        where = f"table {name or '(no name)'}"
        if not (NAME.match(name) and name.endswith("_v")):
            out.append(f"{where}: name must be snake_case ending in _v")
        if name in BASE_TABLES:
            out.append(f"{where}: name clashes with a base table")
        if t.get("window") not in windows:
            out.append(f"{where}: window {t.get('window')!r} is not defined")
        m = APPROVED_FROM.match(t.get("from", ""))
        if not m or m.group(1) == m.group(3):
            out.append(f"{where}: from may only join {', '.join(BASE_TABLES)} on complaint_id")
        for k in ("rows", "leaves_out"):
            if not str(t.get(k, "")).strip():
                out.append(f"{where}: {k} is empty")
        cols = t.get("column", [])
        cnames = [c.get("name", "") for c in cols]
        if not cols:
            out.append(f"{where}: no columns")
        if "complaint_id" not in cnames:
            out.append(f"{where}: needs complaint_id")
        for cname in sorted({n for n in cnames if cnames.count(n) > 1}):
            out.append(f"{where}: column {cname} appears more than once")
        for c in cols:
            if not NAME.match(c.get("name", "")):
                out.append(f"{where}: column {c.get('name')!r} is not snake_case")
            if not str(c.get("description", "")).strip():
                out.append(f"{where}: column {c.get('name')} has no description")

    by_table = {t.get("name"): {c.get("name") for c in t.get("column", [])} for t in tables}
    mnames = [m.get("name", "") for m in layer.get("metric", [])]
    for name in sorted({n for n in mnames if mnames.count(n) > 1}):
        out.append(f"metric {name} appears more than once")
    for m in layer.get("metric", []):
        where = f"metric {m.get('name') or '(no name)'}"
        if not NAME.match(m.get("name", "")):
            out.append(f"{where}: name is not snake_case")
        if m.get("table") not in by_table:
            out.append(f"{where}: table {m.get('table')!r} is not in the layer")
        if m.get("kind") not in KINDS:
            out.append(f"{where}: kind must be one of {', '.join(KINDS)}")
        if m.get("kind") == "rate" and not (m.get("numerator") and m.get("denominator")):
            out.append(f"{where}: a rate needs numerator and denominator")
        if m.get("kind") in ("count", "share") and (m.get("numerator") or m.get("denominator")):
            out.append(f"{where}: only a rate takes numerator and denominator")
        if not str(m.get("description", "")).strip():
            out.append(f"{where}: description is empty")
    return out


def load(path=LAYER_PATH):
    with open(path, "rb") as f:
        layer = tomllib.load(f)
    found = problems(layer)
    if found:
        raise LayerError(f"{path}:\n  " + "\n  ".join(found))
    layer["sha256"] = file_sha256(path)
    layer["tables"] = {t["name"]: t for t in layer["table"]}
    layer["metrics"] = {m["name"]: m for m in layer.get("metric", [])}
    return layer


def table_sql(layer, t, schema=""):
    """SELECT for one layer table, from the base tables (in schema, if given)."""
    w = layer["window"][t["window"]]
    source = t["from"]
    if schema:
        source = re.sub(r"\b(complaints|narratives)\b", schema + r".\1", source)
    cols = ",\n  ".join(f"{c['sql']} AS {c['name']}" if c.get("sql") else c["name"]
                        for c in t["column"])
    where = f"date_received BETWEEN '{w['from']}' AND '{w['to']}'"
    if t.get("where"):
        where += f" AND ({t['where']})"
    return f"SELECT\n  {cols}\nFROM {source}\nWHERE {where}\nORDER BY complaint_id"


# ---------- building data/agent.sqlite ----------

def uri(path, read_only=True):
    """file: URI for path. Absolute and percent-encoded, so a Windows path with
    spaces works, and ATTACH of a file: URI works on this connection."""
    return Path(path).resolve().as_uri() + ("?mode=ro" if read_only else "")


def connect_source(path):
    if not os.path.exists(path):
        raise LayerError(f"{path} not found. Run python -m ingest.build_db")
    conn = sqlite3.connect(uri(path), uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def build(source_path=sources.DB_PATH, agent_path=AGENT_DB_PATH, layer_path=LAYER_PATH,
          force=False):
    layer = load(layer_path)
    if os.path.exists(agent_path) and not force:
        raise LayerError(f"{agent_path} exists. Use --force to rebuild it")
    tmp_path = agent_path + ".building"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    src = connect_source(source_path)
    source_info = dict(src.execute("SELECT key, value FROM build_info"))
    src.close()

    os.makedirs(os.path.dirname(agent_path) or ".", exist_ok=True)
    conn = sqlite3.connect(uri(tmp_path, read_only=False), uri=True)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("ATTACH DATABASE ? AS src", (uri(source_path),))
        counts = {}
        for t in layer["table"]:
            conn.execute(f"CREATE TABLE {t['name']} AS {table_sql(layer, t, 'src')}")
            conn.execute(f"CREATE UNIQUE INDEX idx_{t['name']}_id ON {t['name']} (complaint_id)")
            conn.execute(f"CREATE INDEX idx_{t['name']}_pm ON {t['name']} (product, month)")
            counts[t["name"]] = conn.execute(f"SELECT COUNT(*) FROM {t['name']}").fetchone()[0]
            print(f"  {t['name']}: {counts[t['name']]:,} rows")
        conn.commit()
        conn.execute("DETACH DATABASE src")
        conn.execute("CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        info = {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_commit": git_commit(),
            "layer_version": layer["version"],
            "layer_sha256": layer["sha256"],
            "source_built_at": source_info.get("built_at", ""),
            "source_loader_commit": source_info.get("loader_commit", ""),
            **{f"rows_{name}": str(n) for name, n in counts.items()},
        }
        conn.executemany("INSERT INTO build_info VALUES (?, ?)", info.items())
        conn.commit()
    except BaseException:
        conn.close()
        os.remove(tmp_path)
        raise
    conn.close()
    os.replace(tmp_path, agent_path)
    return info


# ---------- the agent's connection ----------

class Guard:
    """SQLite authorizer for the agent's connection.

    Not an allowlist of table names: SQLite reports a read of a CTE's own
    name (for COUNT(*) over a CTE), so that would refuse ordinary SQL. The
    window guarantee comes from what data/agent.sqlite contains, not from here.
    last_denial says what was refused, so run_sql can give the agent a message
    it can act on instead of SQLite's bare error.
    """

    def __init__(self, tables):
        self.tables = sorted(tables)
        self.last_denial = None

    def __call__(self, action, arg1, arg2, db_name, inner):
        if action not in ALLOWED_ACTIONS:
            what = ACTION_NAMES.get(action, f"action {action}")
            self.last_denial = f"only SELECT queries are allowed, not {what}"
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and (arg1 in HIDDEN_TABLES
                                              or (arg1 or "").startswith("sqlite_")):
            self.last_denial = (f"{arg1} is not part of the semantic layer; "
                                f"the tables are {', '.join(self.tables)}")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK


def connect(agent_path=AGENT_DB_PATH, layer_path=LAYER_PATH):
    """Read-only connection to the agent's database, guarded. Returns (conn, guard)."""
    layer = load(layer_path)
    if not os.path.exists(agent_path):
        raise LayerError(f"{agent_path} not found. Run python -m agent.semantic build")
    conn = sqlite3.connect(uri(agent_path), uri=True)
    conn.execute("PRAGMA query_only = ON")
    info = dict(conn.execute("SELECT key, value FROM build_info"))
    if info.get("layer_sha256") != layer["sha256"]:
        conn.close()
        raise LayerError(
            f"{agent_path} was built from layer {info.get('layer_version')} "
            f"({info.get('layer_sha256', '')[:12]}), but {layer_path} is now "
            f"{layer['version']} ({layer['sha256'][:12]}). "
            "Run python -m agent.semantic build --force")
    guard = Guard(layer["tables"])
    conn.set_authorizer(guard)
    return conn, guard


# ---------- metrics ----------

def metric_sql(layer, name, by=(), filters=None):
    """SQL and parameters for a defined metric.

    by: columns of the metric's table to group by.
    filters: {column: value or list of values}, combined with AND.
    Returns (sql, params). Result columns: the by columns, numerator,
    denominator (None for a count), value (count, or a percentage).
    """
    if name not in layer["metrics"]:
        raise LayerError(f"unknown metric {name}")
    m = layer["metrics"][name]
    cols = {c["name"] for c in layer["tables"][m["table"]]["column"]}
    filters = filters or {}
    for col in list(by) + list(filters):
        if col not in cols:
            raise LayerError(f"{m['table']} has no column {col}")
    if m["kind"] == "share" and not by:
        raise LayerError(f"{name} is a share, so it needs at least one by column")

    clauses, params = [], []
    if m.get("where"):
        clauses.append(f"({m['where']})")
    for col, value in filters.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        clauses.append(f"{col} IN ({', '.join('?' * len(values))})")
        params += list(values)

    if m["kind"] == "count":
        num, den, value = "COUNT(*)", "NULL", "COUNT(*)"
    elif m["kind"] == "share":
        num, den = "COUNT(*)", "SUM(COUNT(*)) OVER ()"
        value = f"100.0 * {num} / {den}"
    else:
        num, den = m["numerator"], m["denominator"]
        value = f"100.0 * {num} / {den}"
    select = list(by) + [f"{num} AS numerator", f"{den} AS denominator", f"{value} AS value"]
    sql = f"SELECT {', '.join(select)} FROM {m['table']}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if by:
        sql += f" GROUP BY {', '.join(by)} ORDER BY {', '.join(by)}"
    return sql, params


def run_metric(conn, layer, name, by=(), filters=None):
    sql, params = metric_sql(layer, name, by, filters)
    cur = conn.execute(sql, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


# ---------- what the agent is shown ----------

def describe(layer):
    """The layer as text, for describe_tables."""
    out = [f"Semantic layer {layer['version']}", "", layer["data_note"].strip(), "", "Windows:"]
    for wname, w in layer["window"].items():
        out.append(f"- {wname}: received {w['from']} to {w['to']}. {w['why'].strip()}")
    for t in layer["table"]:
        w = layer["window"][t["window"]]
        out += ["", f"Table {t['name']} (window {t['window']}, {w['from']} to {w['to']})",
                t["rows"].strip(), f"Leaves out: {t['leaves_out'].strip()}", "Columns:"]
        out += [f"- {c['name']}: {c['description'].strip()}" for c in t["column"]]
    out += ["", "Metrics (use these definitions when a request names one):"]
    for m in layer["metrics"].values():
        if m["kind"] == "rate":
            formula = f"100.0 * {m['numerator']} / {m['denominator']}"
        elif m["kind"] == "share":
            formula = "100.0 * COUNT(*) / SUM(COUNT(*)) OVER () per group"
        else:
            formula = "COUNT(*)"
        extra = f" WHERE {m['where']}" if m.get("where") else ""
        out.append(f"- {m['name']}: {m['description'].strip()} "
                   f"{formula} FROM {m['table']}{extra}")
    return "\n".join(out)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "check"
    try:
        if cmd == "check":
            layer = load()
            print(f"{LAYER_PATH}: {layer['version']}, sha256 {layer['sha256']}")
            print(f"tables {', '.join(layer['tables'])}")
            print(f"metrics {', '.join(layer['metrics'])}")
        elif cmd == "build":
            info = build(force="--force" in argv)
            print(f"Built {AGENT_DB_PATH} from layer {info['layer_version']} "
                  f"at code commit {info['code_commit']}")
        elif cmd == "show":
            print(describe(load()))
        elif cmd == "metrics":
            layer = load()
            conn, _ = connect()
            for name, m in layer["metrics"].items():
                by = ("product",)
                print(f"\n== {name} ({m['table']}, by product)")
                for r in run_metric(conn, layer, name, by):
                    den = "" if r["denominator"] is None else f" / {r['denominator']:,}"
                    val = f"{r['value']:,}" if m["kind"] == "count" else f"{r['value']:.2f}%"
                    print(f"  {r['product']:<28} {r['numerator']:,}{den} = {val}")
            conn.close()
        else:
            print(__doc__)
            return 2
    except LayerError as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
