"""The agent's tools. This file has the first two: describe_tables and run_sql.
tag_root_causes is in agent/tagger.py, clean_and_chart and build_report in
agent/report.py, send_report in agent/send.py.

Each tool returns a dict for code (the graph, the eval harness, the run log)
and a text form for the model (as_text). Nothing a tool returns to the model
contains narrative text: the agent's connection can't read it (layer-2).

run_sql runs one SELECT on the guarded, read-only connection with three caps:
  rows   at most max_rows rows are returned; the total is still counted
  cells  a text value longer than max_cell characters is cut
  time   a query running longer than max_seconds is stopped
Errors come back as messages the agent can act on, never as exceptions, so the
graph can hand them back to the model to correct its SQL.

Usage (for trying a query by hand on the real database):
  python -m agent.tools describe
  python -m agent.tools sql "SELECT product, COUNT(*) FROM complaints_v GROUP BY product"
  python -m agent.tools textcheck     counts and timings only, never text
"""
import random
import sqlite3
import sys
import time

from agent import semantic as sem
from agent import text as textmod

MAX_ROWS = 200
MAX_CELL = 300
MAX_SECONDS = 5.0
PROGRESS_STEPS = 10_000      # SQLite VM steps between time checks
MAX_SQL = 5_000              # characters in one query
PROBE_SIZES = (400, 700)     # complaints in the two data-check samples, at most
PROBE_SEED = 20261111
PROBE_SECONDS = 1.0          # per query per sample
PROBE_ROWS = 5_000


def describe_tables(layer):
    """The layer as the agent sees it: tables, columns, windows, metrics, text_has."""
    return {"ok": True, "text": sem.describe(layer)}


def _cell(value, max_cell):
    """(shown value, was it cut). Floats are rounded to 4 places for display."""
    if isinstance(value, float):
        return round(value, 4), False
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>", False
    if isinstance(value, str) and len(value) > max_cell:
        return value[:max_cell] + "…", True
    return value, False


class DataProbe:
    """Two small samples of the agent's database, to tell which result columns
    depend on the data (Prit, S6: closes the number check's typed-in-numbers hole).

    A query's numeric column whose set of values is the same on the real data
    and on both samples doesn't depend on the data (e.g. ROUND(15.6 + 14.6, 1),
    or a constant beside a GROUP BY), so the report's number check ignores it.
    Samples: about a third and two thirds of the complaints (at most
    PROBE_SIZES), the same complaint IDs in every table so joins still work.
    text_has is replaced by a stand-in that depends on the complaint ID. If a
    query can't run on the samples, nothing is marked (the check stays as before)."""

    def __init__(self, conn, layer, sizes=None, seed=None):
        sizes = sizes or PROBE_SIZES
        rng = random.Random(PROBE_SEED if seed is None else seed)
        ids = [r[0] for r in conn.execute(
            "SELECT complaint_id FROM complaints_v ORDER BY complaint_id")]
        self.samples = []
        for part, cap in zip((3, 1.5), sizes):
            chosen = sorted(rng.sample(ids, max(1, min(cap, int(len(ids) / part)))))
            mem = sqlite3.connect(":memory:", check_same_thread=False)
            for name, table in layer["tables"].items():
                cols = [c["name"] for c in table["column"]]
                mem.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
                for i in range(0, len(chosen), 900):
                    chunk = chosen[i:i + 900]
                    rows = conn.execute(
                        f"SELECT {', '.join(cols)} FROM {name} WHERE complaint_id IN "
                        f"({', '.join('?' * len(chunk))})", chunk).fetchall()
                    mem.executemany(f"INSERT INTO {name} VALUES "
                                    f"({', '.join('?' * len(cols))})", rows)
            mem.create_function("text_has", 2, _stand_in_text_has, deterministic=True)
            mem.execute("PRAGMA query_only = ON")
            self.samples.append(mem)

    def independent(self, sql, columns, rows, row_count):
        """Names of the columns that don't depend on the data, plus '#rows' if
        the number of rows doesn't either."""
        real = _numbers_by_column(rows, len(columns))
        found = []
        for mem in self.samples:
            deadline = time.monotonic() + PROBE_SECONDS
            try:
                mem.set_progress_handler(lambda: int(time.monotonic() > deadline),
                                         PROGRESS_STEPS)
                cur = mem.execute(sql)
                if not cur.description or len(cur.description) != len(columns):
                    return set()
                got = cur.fetchmany(PROBE_ROWS)
                mem.set_progress_handler(None, 0)
            except sqlite3.Error:
                return set()
            found.append((_numbers_by_column(got, len(columns)), len(got)))
        out = {name for i, name in enumerate(columns)
               if real[i] and all(f[0][i] == real[i] for f in found)}
        if row_count is not None and all(n == row_count for _, n in found):
            out.add("#rows")
        return out

    def close(self):
        for mem in self.samples:
            mem.close()


def _stand_in_text_has(complaint_id, phrase):
    if not isinstance(complaint_id, int) or not isinstance(phrase, str):
        return None
    return (complaint_id + sum(map(ord, phrase))) % 2


def _numbers_by_column(rows, n):
    sets = [set() for _ in range(n)]
    for row in rows:
        for i, v in enumerate(row[:n]):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sets[i].add(round(float(v), 9))
    return sets


def _error(kind, message, started):
    return {"ok": False, "error_kind": kind, "error": message,
            "seconds": round(time.monotonic() - started, 3)}


def run_sql(conn, guard, sql, max_rows=MAX_ROWS, max_cell=MAX_CELL, max_seconds=MAX_SECONDS,
            store=None, probe=None):
    """Run one SELECT. Returns a dict:

    ok, columns, rows (at most max_rows, cells capped), row_count (all rows the
    query returned, or None when counting was stopped by the time limit),
    truncated (more rows than shown), cells_cut, seconds. On failure: ok False,
    error_kind (empty, too_long, one_statement, refused, timeout, text_has,
    sql) and error, a message saying what to change.

    store: the run's agent.report.RunStore. A successful result is saved there
    and gets a handle (rows_1, rows_2, ...), so charts and the report use the
    numbers the query returned, never numbers retyped by the model.
    probe: the run's DataProbe. Columns that don't depend on the data are
    recorded with the result (independent) and named in its text."""
    started = time.monotonic()
    if not isinstance(sql, str) or not sql.strip():
        return _error("empty", "The query is empty. Send one SELECT statement.", started)
    if len(sql) > MAX_SQL:
        return _error("too_long", f"The query is over {MAX_SQL:,} characters. Shorten it.", started)

    deadline = started + max_seconds
    timed_out = []

    def progress():
        if time.monotonic() > deadline:
            timed_out.append(True)
            return 1
        return 0

    guard.last_denial = None
    conn.set_progress_handler(progress, PROGRESS_STEPS)
    rows, count, complete = [], 0, True
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        try:
            while True:
                batch = cur.fetchmany(500)
                if not batch:
                    break
                count += len(batch)
                room = max_rows - len(rows)
                if room > 0:
                    rows.extend(batch[:room])
        except sqlite3.OperationalError:
            if not timed_out:
                raise
            complete = False          # rows shown so far are fine; the total isn't known
    except sqlite3.ProgrammingError as e:
        if "one statement" in str(e):
            return _error("one_statement", "Send one SELECT statement at a time.", started)
        return _error("sql", f"SQLite refused the query: {e}", started)
    except sqlite3.DatabaseError as e:
        if timed_out:
            return _error("timeout", (
                f"The query ran over {max_seconds:g} seconds and was stopped. Narrow it: "
                "filter by product or month first, join on complaint_id only, and avoid "
                "joining a table to itself."), started)
        if guard.last_denial and guard.last_denial.startswith("text_has"):
            return _error("text_has", guard.last_denial + ".", started)
        if guard.last_denial:
            return _error("refused", f"Not allowed: {guard.last_denial}.", started)
        message = str(e)
        hint = ""
        if message.startswith(("no such table", "no such column")):
            hint = " Call describe_tables to see the tables and their columns."
        return _error("sql", f"SQLite error: {message}.{hint}", started)
    finally:
        conn.set_progress_handler(None, 0)

    shown, cut = [], 0
    for row in rows:
        out = []
        for v in row:
            value, was_cut = _cell(v, max_cell)
            cut += was_cut
            out.append(value)
        shown.append(out)
    result = {
        "ok": True,
        "columns": columns,
        "rows": shown,
        "row_count": count if complete else None,
        "truncated": count > len(shown) or not complete,
        "cells_cut": cut,
        "seconds": round(time.monotonic() - started, 3),
    }
    if probe is not None:
        result["independent"] = sorted(probe.independent(sql, columns, rows, result["row_count"]))
    if store is not None:
        result["handle"] = store.add_table(sql, result)
    return result


def _fmt(value):
    if value is None:
        return "NULL"
    return str(value).replace("\n", " ").replace("|", "/")


def as_text(result):
    """The result as the model sees it."""
    if "text" in result:
        return result["text"]
    if not result["ok"]:
        return f"ERROR: {result['error']}"
    lines = [" | ".join(result["columns"])]
    lines += [" | ".join(_fmt(v) for v in row) for row in result["rows"]]
    n = len(result["rows"])
    if result["row_count"] is None:
        foot = (f"({n} rows shown; the time limit stopped the count, so there are more. "
                "Add a filter or aggregate.)")
    elif result["truncated"]:
        foot = (f"({n} of {result['row_count']:,} rows shown. Aggregate with GROUP BY or "
                "add LIMIT and ORDER BY to see the rows you need.)")
    else:
        foot = f"({n} row{'s' if n != 1 else ''})"
    if result["cells_cut"]:
        foot += f" {result['cells_cut']} long values were cut."
    if result.get("handle"):
        foot += f" Saved as {result['handle']} for clean_and_chart and the report."
    fixed = [c for c in result.get("independent", ()) if c != "#rows"]
    if fixed:
        foot += (f" Column(s) {', '.join(fixed)} don't depend on the data (same values on "
                 "samples of the database), so their numbers can't be used in findings.")
    return "\n".join(lines + [foot])


def text_check():
    """Counts of b'...' narratives and how readable() handles them, then
    text_has timings on the agent's connection. Prints no narrative text."""
    conn = sem.connect_text()
    wrapped = decoded = 0
    for (text,) in conn.execute(f"SELECT narrative FROM {textmod.TEXT_TABLE}"):
        if textmod.is_bytes_literal(text):
            wrapped += 1
            decoded += textmod.decode_bytes_literal(text) is not None
    total = conn.execute(f"SELECT COUNT(*) FROM {textmod.TEXT_TABLE}").fetchone()[0]
    conn.close()
    print(f"narratives {total:,}; stored as b'...' {wrapped:,} "
          f"(valid bytes literal {decoded:,}, escapes replaced instead {wrapped - decoded:,})")
    conn, guard = sem.connect()
    for phrase in ("late fee", "zelle", "refund*", "late fee"):
        r = run_sql(conn, guard, "SELECT product, SUM(text_has(complaint_id, "
                    f"'{phrase}')) AS mentions, COUNT(*) AS narratives FROM narratives_v "
                    "GROUP BY product ORDER BY product")
        print(f"\n{phrase!r} ({r['seconds']} s)")
        print(as_text(r))
    r = run_sql(conn, guard, "SELECT COUNT(*) FROM complaints_v "
                "WHERE text_has(complaint_id, 'overdraft') = 1")
    print(f"\n'overdraft' over all 125,220 complaints ({r['seconds']} s)")
    print(as_text(r))
    conn.close()


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        layer = sem.load()
        if cmd == "describe":
            print(as_text(describe_tables(layer)))
        elif cmd == "textcheck":
            text_check()
        elif cmd == "sql" and len(argv) == 3:
            conn, guard = sem.connect()
            result = run_sql(conn, guard, argv[2])
            print(as_text(result))
            print(f"[{result['seconds']} s]")
            conn.close()
        else:
            print(__doc__)
            return 2
    except sem.LayerError as e:
        print(f"ERROR {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
