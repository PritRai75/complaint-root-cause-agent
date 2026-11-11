"""The report tools: clean_and_chart and build_report, and the run's store.

Decisions (session 6, Prit): charts are matplotlib PNGs with a table of the same
numbers under each one; python-docx writes the Word file; Pandas shapes the
tables. build_report refuses findings that contain a number not found in this
run's results. Quotes: one for each of the top 3 root causes, picked by code.

Where numbers come from. Every number in the report comes from a result saved
in the run's store: a query (rows_N, saved by run_sql), a tag set (tags_N, saved
by tag_root_causes) or a chart made from them (chart_N). The model refers to
them by handle and never retypes them. The model writes the title and the
findings; code checks every number in them against the store (rounding
allowed, see supported()) and refuses legal conclusions. Numbers from query
columns that don't depend on the data (tools.DataProbe: e.g. numbers typed
into the SQL) don't count, and a chart from rows_N adds no numbers of its own
(they are its table's). It does not check
whether a finding says a figure rose when it fell: that is left to the eval
and listed as a limitation.

Narrative text. The planning model never sees it (decision 15). build_report
reads the quotes through the text connection and puts them in the Word file;
what it returns to the model is counts only.

Chart style (dataviz method, default light palette): one hue per series in
fixed order, thin bars with the value at the tip, recessive axes, no chart
title inside the image (the report heading names it), a legend for two or more
series. The image's alt text lists every value, and the table under the chart
repeats them as text.

Usage (real data, on the laptop):
  python -m agent.report sample              queries, charts and report; no model calls
  python -m agent.report sample --tags 20    also tags 20 card narratives on Bedrock
The report goes to scratch/sample-report.docx (gitignored).
"""
import hashlib
import io
import math
import random
import re
import textwrap
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from agent import tagger as tg  # noqa: E402
from agent import text as textmod  # noqa: E402

# chart look
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")     # dataviz default, slots 1 to 3
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
DPI = 200
WIDTH_IN = 6.5
MAX_BARS = 12
MAX_SERIES = 3
MAX_CHARTS = 6
LABEL_WRAP = 30          # characters per line of a bar label

# report limits
MAX_TITLE = 120
MAX_FINDINGS = 8
MAX_FINDING = 500
QUOTE_CAUSES = 3
QUOTE_MIN = 150          # characters; shorter narratives aren't quoted
QUOTE_MAX = 400
QUOTE_SEED = 20261111
AGENT = "complaint-root-cause-agent"

MONTH = re.compile(r"^\d{4}-\d{2}$")
CAVEAT = ("These findings describe what consumers reported to the CFPB. Complaint "
          "narratives are not verified, and complaints are not a representative sample "
          "of all customers. Nothing in this report is a finding that a company broke a "
          "rule or a law.")
QUOTE_NOTE = ("One complaint for each of the most common root causes, picked by code at "
              "random among those tagged with it, never by company. Excerpts are as "
              "published: company names and any claims about the law are the consumer's "
              "own words, and XXXX marks the CFPB's redactions.")
LEGAL = re.compile(r"\b(illegal\w*|unlawful\w*|violat\w*|broke the law|breach\w* of the law)\b",
                   re.IGNORECASE)


# ---------- the run's store ----------

class RunStore(tg.TagStore):
    """Everything one agent run made, by handle: query results (rows_N), tag
    sets (tags_N, from TagStore), charts (chart_N) and the report. The question
    is set by code from the request, never by the model."""

    def __init__(self, question=""):
        super().__init__()
        self.question = question
        self.tables = {}
        self.charts = {}
        self.report = None
        self.send_attempts = 0
        self.sent = None

    def add_table(self, sql, result):
        handle = f"rows_{len(self.tables) + 1}"
        self.tables[handle] = {"sql": sql, "columns": list(result["columns"]),
                               "rows": [list(r) for r in result["rows"]],
                               "row_count": result["row_count"],
                               "truncated": result["truncated"],
                               "independent": list(result.get("independent", ()))}
        return handle


def _err(kind, message):
    return {"ok": False, "error_kind": kind, "error": message}


_TAX = None


def taxonomy():
    global _TAX
    if _TAX is None:
        from eval import taxonomy as taxmod
        _TAX = taxmod.load()
    return _TAX


def cause_names():
    return {c["code"]: c["name"].strip() for c in taxonomy()["cause"]}


# ---------- formatting ----------

def fmt_value(value, unit):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if unit == "percent":
        return f"{value:.1f}%"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


# ---------- clean_and_chart ----------

def _frame_from_tags(store, handle, y):
    """Counts or shares of root causes for a tag set, in precedence order.
    third_party, other and unclear are left out: shares are over root causes."""
    if y not in ("share", "count"):
        return None, _err("args", f"For a tag set, y is 'share' or 'count', not {y!r}.")
    info = store.info.get(handle, {})
    counts = info.get("counts") or {}
    shares = info.get("shares") or {}
    if not counts:
        counted = {}
        for _, code in store.sets[handle]:
            counted[code] = counted.get(code, 0) + 1
        counts = {c: counted.get(c, 0) for c in taxonomy()["precedence"]}
        base = sum(n for c, n in counts.items() if c not in tg.NOT_ROOT_CAUSES)
        shares = {c: (round(100.0 * n / base, 1) if base else None) for c, n in counts.items()}
    names = cause_names()
    rows = [(names.get(c, c), counts[c] if y == "count" else shares.get(c))
            for c in counts if c not in tg.NOT_ROOT_CAUSES]
    frame = pd.DataFrame(rows, columns=["root cause", y])
    unit = "percent" if y == "share" else "count"
    return (frame, "root cause", y, None, unit), None


def _frame_from_rows(store, handle, x, y, series, unit):
    t = store.tables[handle]
    if t["truncated"]:
        total = "more" if t["row_count"] is None else f"{t['row_count']:,}"
        return None, _err("too_many_rows", (
            f"{handle} has {total} rows but only {len(t['rows'])} were returned. Aggregate "
            "in SQL (GROUP BY) so the whole result fits, then chart the new handle."))
    cols = t["columns"]
    roles = [r for r in (x, y, series) if r is not None]
    if len(set(roles)) != len(roles):
        return None, _err("args", "x, y and series must be different columns.")
    for name, role in ((x, "x"), (y, "y"), (series, "series")):
        if name is None and role == "series":
            continue
        if name not in cols:
            return None, _err("args", f"{handle} has no column {name!r} for {role}. "
                                      f"Its columns: {', '.join(cols)}.")
        if cols.count(name) > 1:
            return None, _err("args", f"{handle} has two columns named {name!r}. Rename one "
                                      "with AS in the query.")
    if unit not in ("count", "percent"):
        return None, _err("args", f"unit is 'count' or 'percent', not {unit!r}.")
    keep = [x, y] + ([series] if series else [])
    frame = pd.DataFrame(t["rows"], columns=cols)[keep].copy()
    return (frame, x, y, series, unit), None


def _clean(frame, x, y, series, kind):
    """Checks and shapes the frame. Returns (frame, notes) or (None, error)."""
    notes = []
    raw = frame[y]
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = numeric.isna() & raw.notna()
    if bad.any():
        return None, _err("not_numeric", f"Column {y!r} has values that aren't numbers "
                                         f"(e.g. {raw[bad].iloc[0]!r}). Chart a numeric column.")
    frame[y] = numeric.astype(float)
    empty = frame[y].isna()
    if empty.any():
        notes.append(f"{int(empty.sum())} row(s) with no value left out.")
        frame = frame[~empty].copy()
    if frame.empty:
        return None, _err("empty", "Nothing to chart: no rows with a value.")
    frame[x] = frame[x].map(lambda v: "(blank)" if v is None or v == "" else str(v))
    if series:
        frame[series] = frame[series].map(lambda v: "(blank)" if v is None or v == ""
                                          else str(v))
        n = frame[series].nunique()
        if n > MAX_SERIES:
            return None, _err("too_many_series", (
                f"{n} series in {series!r}; at most {MAX_SERIES}. Filter to the main ones in "
                "SQL or make separate charts."))
    keys = [x] + ([series] if series else [])
    if frame.duplicated(keys).any():
        return None, _err("duplicates", (
            f"Some {' and '.join(repr(k) for k in keys)} values appear more than once. "
            "Group by them in SQL so each appears once."))
    months = bool(frame[x].map(lambda v: bool(MONTH.match(v))).all())
    if kind == "line":
        if not months:
            return None, _err("args", "A line chart needs months (YYYY-MM) on x. Use a bar "
                                      "chart for categories.")
        if frame[x].nunique() < 2:
            return None, _err("args", "A line chart needs at least two months.")
        frame = frame.sort_values(keys).reset_index(drop=True)
    else:
        if months:
            frame = frame.sort_values(keys).reset_index(drop=True)
        else:
            totals = frame.groupby(x)[y].sum().sort_values(ascending=False, kind="stable")
            if len(totals) > MAX_BARS:
                notes.append(f"Top {MAX_BARS} of {len(totals)} categories shown.")
                totals = totals.iloc[:MAX_BARS]
            order = {k: i for i, k in enumerate(totals.index)}
            frame = frame[frame[x].isin(order)].copy()
            frame["_o"] = frame[x].map(order)
            frame = frame.sort_values(["_o"] + ([series] if series else []), kind="stable")
            frame = frame.drop(columns="_o").reset_index(drop=True)
    return frame, notes


def month_range(first, last):
    """Every YYYY-MM from first to last, inclusive."""
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _value_axis(axis, unit, nbins="auto"):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    if unit == "count":
        axis.set_major_locator(MaxNLocator(nbins=nbins, integer=True))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        axis.set_major_locator(MaxNLocator(nbins=nbins))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))


def wrap(label, width=LABEL_WRAP):
    """A long category name over at most two lines, for the chart only (the
    table and the alt text keep it whole)."""
    lines = textwrap.wrap(label, width) or [label]
    if len(lines) > 2:
        lines = [lines[0], textwrap.shorten(" ".join(lines[1:]), width, placeholder=" …")]
    return "\n".join(lines)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_axisbelow(True)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def draw(frame, x, y, series, kind, unit, y_label):
    """PNG bytes. Bars are horizontal (long category names stay readable)."""
    label = (lambda v: fmt_value(v, unit))
    groups = [None] if not series else list(dict.fromkeys(frame[series]))
    if kind == "line":
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
        months = month_range(frame[x].min(), frame[x].max())
        for i, g in enumerate(groups):
            part = frame if g is None else frame[frame[series] == g]
            values = part.set_index(x)[y].reindex(months)    # NaN where a month has no row
            ax.plot(range(len(months)), values.to_numpy(), color=SERIES_COLORS[i],
                    linewidth=1.5, marker="o", markersize=4, label=g)
            last = values.last_valid_index()
            ax.annotate(label(values[last]), (months.index(last), values[last]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.set_xticks(range(len(months)), months)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_ylabel(y_label, fontsize=8, color=INK_2)
        ax.set_ylim(bottom=min(0, frame[y].min()))
        ax.set_xlim(-0.3, len(months) - 0.4)
        _value_axis(ax.yaxis, unit)
    else:
        cats = list(dict.fromkeys(frame[x]))
        n = len(groups)
        band = 0.8
        thick = min(band / n, 0.5)
        labels = [wrap(c) for c in cats]
        per = 0.28 + (0.08 if any("\n" in s for s in labels) else 0)
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.6 + per * len(cats) * max(1, n * 0.7)))
        top, low = max(frame[y].max(), 0), min(frame[y].min(), 0)
        for i, g in enumerate(groups):
            part = frame if g is None else frame[frame[series] == g]
            pos = [cats.index(c) + (i - (n - 1) / 2) * thick for c in part[x]]
            bars = ax.barh(pos, part[y], height=thick * 0.85, color=SERIES_COLORS[i], label=g)
            ax.bar_label(bars, labels=[label(v) for v in part[y]], padding=3, fontsize=7.5,
                         color=INK)
        ax.set_yticks(range(len(cats)), labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlabel(y_label, fontsize=8, color=INK_2)
        ax.set_xlim(low * 1.15, top * 1.15 if top else 1)
        _value_axis(ax.xaxis, unit, nbins=5)
    _style(ax)
    if series:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    fig.tight_layout()
    return _png(fig)


def clean_and_chart(store, handle, title, kind="bar", x=None, y=None, series=None,
                    unit="count"):
    """The planner's chart tool.

    handle: rows_N (a query result) or tags_N (a tag set). For rows_N give x (the
    category or month column), y (the numeric column), optionally series (a
    column with at most 3 values, e.g. product) and unit ('count' or
    'percent'). For tags_N give y = 'share' (default) or 'count'.
    kind: 'bar' (categories, largest first; months stay in order) or 'line'
    (months on x).
    Returns the cleaned table as text and a chart_N handle for build_report."""
    if not isinstance(title, str) or not title.strip():
        return _err("args", "Give the chart a title.")
    title = " ".join(title.split())[:MAX_TITLE]
    if kind not in ("bar", "line"):
        return _err("args", f"kind is 'bar' or 'line', not {kind!r}.")
    if len(store.charts) >= MAX_CHARTS:
        return _err("too_many_charts", f"A report has at most {MAX_CHARTS} charts.")
    if handle in store.sets:
        if kind != "bar":
            return _err("args", "Root causes are categories: use kind 'bar'.")
        shaped, err = _frame_from_tags(store, handle, y or "share")
    elif handle in store.tables:
        if not x or not y:
            return _err("args", "Give x and y: column names from " + handle + ".")
        shaped, err = _frame_from_rows(store, handle, x, y, series, unit)
    else:
        known = sorted(store.tables) + sorted(store.sets)
        return _err("unknown_handle", f"No result called {handle!r}. Known: "
                                      f"{', '.join(known) or 'none yet'}.")
    if err:
        return err
    frame, x, y, series, unit = shaped
    frame, notes = _clean(frame, x, y, series, kind)
    if frame is None:
        return notes
    y_label = y if unit == "count" else f"{y} (%)"
    png = draw(frame, x, y, series, kind, unit, y_label)

    order = (month_range(frame[x].min(), frame[x].max()) if kind == "line"
             else list(dict.fromkeys(frame[x])))
    if series:
        table = frame.pivot(index=x, columns=series, values=y)
        table = table.reindex(index=order, columns=list(dict.fromkeys(frame[series])))
    else:
        table = frame.set_index(x)[[y]].reindex(order)
    columns = [x] + [str(c) for c in table.columns]
    rows = [[str(idx)] + [fmt_value(v, unit) for v in vals]
            for idx, vals in zip(table.index, table.to_numpy().tolist())]
    if kind == "line" and table.isna().to_numpy().any():
        notes.append("Months with no row show as '-' and break the line.")
    alt = f"{title}. " + "; ".join(
        f"{r[0]}: " + ", ".join(f"{c} {v}" if series else v
                                for c, v in zip(columns[1:], r[1:])) for r in rows)
    chart = f"chart_{len(store.charts) + 1}"
    store.charts[chart] = {"title": title, "png": png, "alt": alt, "kind": kind,
                           "columns": columns, "rows": rows, "notes": notes, "source": handle,
                           "unit": unit, "values": [float(v) for v in frame[y]]}
    lines = [f"{chart}: {title} ({kind} chart from {handle})", " | ".join(columns)]
    lines += [" | ".join(r) for r in rows]
    lines += notes + [f"Saved as {chart} for build_report."]
    return {"ok": True, "handle": chart, "rows": len(rows), "notes": notes,
            "png_sha256": hashlib.sha256(png).hexdigest(), "text": "\n".join(lines)}


# ---------- the number check ----------

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*(?:%|percent\b))?")
DATES = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
YEARS = range(2011, 2027)        # the CFPB database starts in 2011; the data ends in 2026
# Number words in the request count as their digits (decision 37, session 8): a
# question asking for "the three most common" lets the report say "top 3".
NUMBER_WORDS = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve".split(), 1)}
NUMBER_WORD = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)


def numbers_in(text):
    """[(number as written, value, decimals, is_percent)] in text, dates left out."""
    text = DATES.sub(" ", text)
    found = []
    for m in NUMBER.finditer(text):
        whole, frac, pct = m.group(1), m.group(2) or "", m.group(3)
        value = float(whole.replace(",", "") + frac)
        found.append((m.group(0).strip(), value, max(len(frac) - 1, 0), bool(pct)))
    return found


def supported(value, decimals, is_percent, allowed):
    """Is a number written in a finding one of the allowed values, as written?

    Allowed roundings: to the decimals written (26.33 -> 26.3, 26); a percentage
    written for a fraction (0.2633 -> 26.3%); and rounding a large number while
    keeping at least two significant figures (1,239 -> 1,200, not 1,000)."""
    if decimals == 0 and is_percent is False and int(value) in YEARS and value.is_integer():
        return True
    for v in allowed:
        v = abs(v)
        candidates = [v] + ([v * 100] if is_percent and v <= 1 else [])
        for c in candidates:
            if round(c, decimals) == value:
                return True
            if decimals == 0 and value >= 100:
                k = 1
                while value % 10 ** k == 0 and 10 ** (k + 1) <= c:
                    if round(c, -k) == value:
                        return True
                    k += 1
    return False


def allowed_values(store):
    values = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            values.add(float(v))
        elif isinstance(v, str):
            try:
                values.add(float(v.replace(",", "").rstrip("%")))
            except ValueError:
                pass

    for t in store.tables.values():
        skip = set(t.get("independent", ()))
        keep = [i for i, c in enumerate(t["columns"]) if c not in skip]
        for row in t["rows"]:
            for i in keep:
                add(row[i])
        if "#rows" not in skip:
            add(t["row_count"])
    for info in store.info.values():
        for key in ("selected", "with_narrative", "drawn", "root_cause_base"):
            add(info.get(key))
        for d in ("counts", "shares", "status"):
            for v in (info.get(d) or {}).values():
                add(v)
    for c in store.charts.values():          # a rows_N chart repeats its table's numbers
        if c["source"] in store.sets:
            for v in c["values"]:
                add(v)
    for _, value, _, _ in numbers_in(store.question or ""):
        values.add(value)
    for m in NUMBER_WORD.finditer(store.question or ""):
        values.add(float(NUMBER_WORDS[m.group(1).lower()]))
    return values


def check_text(store, texts):
    """Problems with the title and findings: numbers not found in this run's
    results, and legal conclusions. Returns a list of messages (empty if fine)."""
    allowed = allowed_values(store)
    problems = []
    for label, text in texts:
        bad = [w for w, v, d, p in numbers_in(text) if not supported(v, d, p, allowed)]
        if bad:
            problems.append(f"{label}: {', '.join(bad)} not found in this run's results")
        m = LEGAL.search(text)
        if m:
            problems.append(f"{label}: '{m.group(0)}' states a legal conclusion")
    return problems


# ---------- quotes ----------

def excerpt(text, limit=QUOTE_MAX):
    """Text cut to about limit characters at a sentence end, else at a word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    window = text[:limit + 1]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", window) if m.end() >= limit // 2]
    if ends:
        return window[:ends[-1]] + " […]"
    cut = window.rfind(" ")
    return window[:cut if cut > limit // 2 else limit].rstrip(" ,;:") + " […]"


def pick_quotes(store, handle, text_conn, seed=QUOTE_SEED, causes=QUOTE_CAUSES,
                min_chars=QUOTE_MIN):
    """One quote for each of the top `causes` root causes in a tag set.

    Top by count (ties in taxonomy precedence order); third_party, other and
    unclear are never quoted. For each code, the tagged complaint IDs are
    shuffled with a fixed seed and the first narrative of at least min_chars
    characters is quoted. Returns [(code, complaint_id, excerpt)]."""
    precedence = taxonomy()["precedence"]
    counts = {}
    for cid, code in store.sets[handle]:
        if code not in tg.NOT_ROOT_CAUSES:
            counts[code] = counts.get(code, 0) + 1
    top = sorted(counts, key=lambda c: (-counts[c], precedence.index(c)))[:causes]
    quotes = []
    for i, code in enumerate(top):
        ids = sorted(cid for cid, c in store.sets[handle] if c == code)
        random.Random(seed + i).shuffle(ids)
        for cid in ids:
            row = text_conn.execute(f"SELECT narrative FROM {textmod.TEXT_TABLE} "
                                    "WHERE complaint_id = ?", (cid,)).fetchone()
            text = " ".join(textmod.readable(row[0]).split()) if row else ""
            if len(text) >= min_chars:
                quotes.append((code, cid, excerpt(text)))
                break
    return quotes


# ---------- build_report ----------

def _para(doc, text, size=None, color=None, italic=False, bold=False, style=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.italic, run.bold = italic, bold
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
    return p


KEEP_WHOLE_ROWS = 8      # tables up to this many rows never break across pages


def _table(doc, columns, rows, keep_together=False):
    """keep_together: a table of up to KEEP_WHOLE_ROWS rows stays on one page
    with the chart above it and its source line below; a longer one keeps its
    header and first row with the chart, may break after that, and repeats
    its header row on the next page."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    t = doc.add_table(rows=1, cols=len(columns))
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(header)
    t.style = "Table Grid"
    for cell, name in zip(t.rows[0].cells, columns):
        cell.text = ""
        cell.paragraphs[0].add_run(str(name)).bold = True
    for row in rows:
        cells = t.add_row().cells
        for j, (cell, value) in enumerate(zip(cells, row)):
            cell.text = str(value)
            if j:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    whole = len(rows) <= KEEP_WHOLE_ROWS
    for i, row in enumerate(t.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = keep_together and (whole or i == 0)
                for r in p.runs:
                    r.font.size = Pt(9)
    return t


def build_report(store, layer, title, findings, charts=(), tags=None, conn=None,
                 text_conn=None, now=None):
    """The planner's report tool.

    title: one line. findings: 1 to 8 short statements, each worded as what
    complaints report. Every number in the title and findings must appear in a
    result from this run (rows_N, tags_N, chart_N), rounded as written; compute
    differences and ratios in SQL. charts: chart_N handles, in order. tags: the
    tags_N handle for the coverage line and the example complaints.
    conn (the agent's connection) and text_conn (semantic.connect_text()) are
    passed by the graph, not the model.

    Returns counts only; the Word file is kept in store.report."""
    if not isinstance(title, str) or not title.strip():
        return _err("args", "Give the report a title.")
    title = " ".join(title.split())
    if len(title) > MAX_TITLE:
        return _err("args", f"The title is over {MAX_TITLE} characters. Shorten it.")
    if isinstance(findings, str):
        findings = [findings]
    findings = [" ".join(str(f).split()) for f in findings or [] if str(f).strip()]
    if not 1 <= len(findings) <= MAX_FINDINGS:
        return _err("args", f"Give 1 to {MAX_FINDINGS} findings.")
    long = [i + 1 for i, f in enumerate(findings) if len(f) > MAX_FINDING]
    if long:
        return _err("args", f"Finding(s) {long} are over {MAX_FINDING} characters. Shorten them.")
    charts = list(charts or [])
    unknown = [c for c in charts if c not in store.charts]
    if unknown or len(set(charts)) != len(charts):
        return _err("unknown_handle", f"Unknown or repeated chart handles: "
                                      f"{', '.join(unknown) or 'repeats'}. Made so far: "
                                      f"{', '.join(store.charts) or 'none'}.")
    if tags is not None and tags not in store.sets:
        return _err("unknown_handle", f"No tag set called {tags!r}. Made so far: "
                                      f"{', '.join(store.sets) or 'none'}.")
    problems = check_text(store, [("title", title)] +
                          [(f"finding {i + 1}", f) for i, f in enumerate(findings)])
    if problems:
        return _err("unsupported", (
            "The report was not built. " + "; ".join(problems) + ". Use only numbers that "
            "appear in your query, tag or chart results, rounded as written (compute "
            "differences and ratios in SQL first), and word findings as what complaints "
            "report, never as legal conclusions."))

    now = now or datetime.now(timezone.utc)
    quotes = pick_quotes(store, tags, text_conn) if tags and text_conn is not None else []
    meta = {}
    if quotes and conn is not None:
        ids = [cid for _, cid, _ in quotes]
        meta = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT complaint_id, product, month FROM narratives_v WHERE complaint_id IN "
            f"({', '.join('?' * len(ids))})", ids)}
    names = cause_names()

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    doc.core_properties.title = title
    doc.core_properties.author = AGENT
    doc.core_properties.created = now.replace(tzinfo=None)
    doc.core_properties.modified = now.replace(tzinfo=None)
    doc.core_properties.last_modified_by = AGENT

    doc.add_heading(title, level=0)
    if store.question:
        _para(doc, "Question: " + " ".join(store.question.split()), italic=True)
    _para(doc, f"Built {now:%d %b %Y %H:%M} UTC by {AGENT}. Data: "
               + " ".join(layer["data_note"].split()), size=8.5, color=INK_2)

    doc.add_heading("Key findings", level=1)
    for f in findings:
        doc.add_paragraph(f, style="List Bullet")
    _para(doc, CAVEAT, size=9, color=INK_2, italic=True)

    if tags:
        info = store.info.get(tags, {})
        status = info.get("status") or {}
        tagged = status.get("tagged", len(store.sets[tags]))
        selected, with_text = info.get("selected", 0), info.get("with_narrative", 0)
        pool = (f"{with_text:,} selected complaints, all with a published narrative"
                if selected == with_text else
                f"{with_text:,} complaints with a published narrative among the "
                f"{selected:,} selected")
        how = (f"drawn at random (seed {info.get('seed')}) from the {pool}"
               if info.get("sampled") else f"which is every one of the {pool}")
        line = (f"Root causes come from {tagged:,} complaint narratives tagged by "
                f"{tg.MODEL_ID} (prompt {tg.PROMPT_VERSION}), {how}. Shares are over the "
                f"{info.get('root_cause_base', 0):,} narratives with a root cause "
                "(third party, other and unclear left out).")
        skipped = status.get("invalid", 0) + status.get("failed", 0)
        if skipped:
            line += f" {skipped} narrative(s) could not be tagged and are left out."
        _para(doc, line, size=9.5)

    if charts:
        doc.add_heading("Charts", level=1)
    for h in charts:
        c = store.charts[h]
        doc.add_heading(c["title"], level=2)
        shape = doc.add_picture(io.BytesIO(c["png"]), width=Inches(6.0))
        shape._inline.docPr.set("descr", c["alt"][:2000])
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
        _table(doc, c["columns"], c["rows"], keep_together=True)
        src = c["source"]
        note = (f"Source: {src}, query in the method section." if src in store.tables
                else f"Source: {src}, root-cause tags (coverage above).")
        _para(doc, " ".join(c["notes"] + [note]), size=8.5, color=INK_2)

    if tags:
        doc.add_heading("Example complaints", level=1)
        if quotes:
            _para(doc, QUOTE_NOTE, size=9, color=INK_2)
        else:
            _para(doc, "No tagged narrative was long enough to quote.", size=9, color=INK_2)
        for code, cid, text in quotes:
            product, month = meta.get(cid, ("", ""))
            _para(doc, names.get(code, code), bold=True)
            _para(doc, f"Complaint {cid}" + (f", {product}, received {month}" if product
                                             else ""), size=8.5, color=INK_2)
            _para(doc, f"“{text}”", italic=True)

    doc.add_heading("Method", level=1)
    windows = layer.get("window", {})
    for key in ("response_final", "narratives"):
        if key in windows:
            w = windows[key]
            doc.add_paragraph(f"{w['from']} to {w['to']}: " + " ".join(w["why"].split()),
                              style="List Bullet")
    doc.add_paragraph(f"Semantic layer {layer['version']}. Every number in the findings "
                      "was checked by code against this run's query, tag and chart "
                      "results. Whether a finding describes a rise or fall correctly is "
                      "not checked by code.", style="List Bullet")
    used = [c["source"] for c in (store.charts[h] for h in charts)]
    for h, t in store.tables.items():
        if h in used:
            _para(doc, f"{h}:", size=9, bold=True)
            p = _para(doc, " ".join(t["sql"].split()), size=8.5)
            p.runs[0].font.name = "Consolas"

    buf = io.BytesIO()
    doc.save(buf)
    data = buf.getvalue()
    store.report = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest(),
                    "title": title, "findings": findings, "charts": charts, "tags": tags,
                    "quotes": [(code, cid) for code, cid, _ in quotes],
                    "filename": f"complaint-report-{now:%Y%m%dT%H%M%SZ}.docx"}
    text = (f"Report built: {len(findings)} findings, {len(charts)} charts, "
            f"{len(quotes)} example complaints, {len(data) / 1024:.0f} KB. "
            "Call send_report to email it.")
    return {"ok": True, "findings": len(findings), "charts": len(charts),
            "quotes": len(quotes), "bytes": len(data), "text": text}


# ---------- a sample report on the real data (no model writes anything) ----------

SAMPLE_OUT = "scratch/sample-report.docx"


def sample(tags=0, out=SAMPLE_OUT):
    """Runs the tools the way the graph will, with fixed queries and findings
    written by code from the results, and saves the report to `out`. Prints
    only what the planner would see. tags > 0 also tags that many card
    narratives from January to June on Bedrock (cached tags cost nothing)."""
    import os
    import time
    from agent import semantic as sem
    from agent import tools
    started = time.monotonic()
    layer = sem.load()
    conn, guard = sem.connect()
    text_conn = sem.connect_text()
    store = RunStore(question="Sample: complaint volumes, top credit card issues and "
                              "timely response by product.")
    probe = tools.DataProbe(conn, layer)

    def show(result):
        print(result["text"] if "text" in result else tools.as_text(result), "\n")
        return result

    vol = show(tools.run_sql(conn, guard, "SELECT month, product, COUNT(*) AS complaints "
                             "FROM complaints_v GROUP BY month, product", store=store, probe=probe))
    top = show(tools.run_sql(conn, guard, "SELECT issue, COUNT(*) AS complaints FROM "
                             "complaints_v WHERE product = 'Credit card' GROUP BY issue "
                             "ORDER BY complaints DESC LIMIT 10", store=store, probe=probe))
    timely = show(tools.run_sql(conn, guard, "SELECT product, ROUND(100.0 * AVG(timely), 2) "
                                "AS timely_pct FROM responses_v GROUP BY product",
                                store=store, probe=probe))
    charts = [show(clean_and_chart(store, vol["handle"], "Complaints by month", kind="line",
                                   x="month", y="complaints", series="product"))["handle"],
              show(clean_and_chart(store, top["handle"], "Top credit card issues",
                                   x="issue", y="complaints"))["handle"],
              show(clean_and_chart(store, timely["handle"], "Timely response, January to June",
                                   x="product", y="timely_pct", unit="percent"))["handle"]]
    issue, n = top["rows"][0]
    rates = {p: v for p, v in timely["rows"]}
    findings = [f"The most common credit card issue consumers chose is '{issue}', with "
                f"{n:,} complaints.",
                "Companies responded on time to "
                + " and ".join(f"{v:.1f}% of {p.lower()} complaints" for p, v in rates.items())
                + " with a final response."]
    tag_handle = None
    tagger = None
    if tags:
        tagger = tg.Tagger(taxonomy(), text_conn)
        t = show(tg.tag_root_causes(conn, guard, tagger, (
            "SELECT complaint_id FROM narratives_v WHERE product = 'Credit card' "
            "AND month <= '2026-06'"), store, sample=tags))
        tag_handle = t["handle"]
        charts.append(show(clean_and_chart(store, tag_handle, "Root causes, credit card "
                                           "narratives"))["handle"])
    r = show(build_report(store, layer, "Sample report: credit card and checking complaints",
                          findings, charts=charts, tags=tag_handle, conn=conn,
                          text_conn=text_conn))
    if r["ok"]:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as f:
            f.write(store.report["bytes"])
        print(f"Saved {out} ({len(store.report['bytes']):,} bytes, sha256 "
              f"{store.report['sha256'][:12]}) in {time.monotonic() - started:.1f} s")
    if tagger:
        tagger.close()
    probe.close()
    text_conn.close()
    conn.close()
    return 0 if r["ok"] else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "sample":
        tags = int(argv[3]) if len(argv) == 4 and argv[2] == "--tags" else 0
        return sample(tags=tags)
    print("Usage: python -m agent.report sample [--tags N]")
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
