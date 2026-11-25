"""Recompute every BigQuery result from the local database and compare (decision 57).

Usage: python -m analytics.recompute

Reads data/complaints.sqlite (read only) and analytics/fraud_mapping.csv,
computes each question's result with pandas, independently of the SQL in
analytics/sql/, and compares it with the CSV saved from BigQuery in
analytics/results/. Counts and flags must match exactly; a rounded rate must
be within half a unit of its last decimal of the unrounded value. Prints one
line per result and exits 1 on any difference.
"""
import os
import sqlite3
import sys

import pandas as pd

from analytics import mapping
from eval import tagrun

DB_PATH = "data/complaints.sqlite"
RESULTS_DIR = "analytics/results"
MIN_COMPLAINTS = 200
SPIKE_MIN = 20
RESPONSE_TO = "2026-06-30"
NARRATIVE_LAST_MONTH = "2026-07-01"
MONTHS = [f"2026-{m:02d}-01" for m in range(1, 9)]

# result name: (file, key columns, {float column: decimals})
RESULTS = {
    "q0": ("q0_check.csv", [], {}),
    "q1": ("q1_monthly.csv", ["product", "month"], {"fraud_share": 4}),
    "q2": ("q2_company_rates.csv", ["company"],
           {"fraud_rate": 4, "wilson_low": 4, "wilson_high": 4}),
    "q2b": ("q2b_company_rates_by_product.csv", ["product", "company"], {"fraud_rate": 4}),
    "q3": ("q3_spikes.csv", ["company", "month"],
           {"mean_prev3": 3, "sd_used": 3, "threshold": 3, "excess": 3}),
    "q4": ("q4_response.csv", ["product", "is_fraud"],
           {"timely_rate": 4, "relief_rate": 4, "monetary_relief_rate": 4}),
    "q6": ("q6_discover_followup.csv", ["company", "product", "month"], {}),
}


class RecomputeError(Exception):
    pass


def load_complaints(db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise RecomputeError(f"{db_path} not found")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql("SELECT complaint_id, date_received, product, issue, sub_issue, "
                           "company, company_response, timely_response FROM complaints", conn)
    finally:
        conn.close()


def flag(df, fmap):
    """The view 00_flagged_view.sql builds, in pandas."""
    m = pd.DataFrame([{"product": k[0], "issue": k[1], "sub_issue": k[2],
                       "is_fraud": r["fraud"] == "yes", "fraud_kind": r["kind"] or None}
                      for k, r in fmap.items()])
    d = df.copy()
    d["sub_issue"] = d["sub_issue"].fillna("")
    out = d.merge(m, on=["product", "issue", "sub_issue"], how="left", validate="many_to_one")
    if out["is_fraud"].isna().any():
        bad = out[out["is_fraud"].isna()].iloc[0]
        raise RecomputeError(f"complaint {bad['complaint_id']} has no mapping row")
    out["is_fraud"] = out["is_fraud"].astype(bool)
    out["month"] = out["date_received"].str[:7] + "-01"
    return out


def q0(f, n_table):
    return pd.DataFrame([{
        "complaints": n_table, "joined": len(f), "distinct_ids": f["complaint_id"].nunique(),
        "fraud": int(f["is_fraud"].sum()),
        "unauthorized_transaction": int((f["fraud_kind"] == "unauthorized_transaction").sum()),
        "new_account": int((f["fraud_kind"] == "new_account").sum())}])


def q1(f):
    g = f.groupby(["month", "product"]).agg(
        complaints=("complaint_id", "size"), fraud=("is_fraud", "sum"),
        unauthorized_transaction=("fraud_kind", lambda s: (s == "unauthorized_transaction").sum()),
        new_account=("fraud_kind", lambda s: (s == "new_account").sum())).reset_index()
    g["fraud_share"] = g["fraud"] / g["complaints"]
    g = g.sort_values(["product", "month"])
    g["fraud_change"] = g.groupby("product")["fraud"].diff()
    return g


def wilson_cols(k, n):
    lo, hi = zip(*(tagrun.wilson(int(a), int(b)) for a, b in zip(k, n)))
    return list(lo), list(hi)


def q2(f):
    g = f.groupby("company").agg(complaints=("complaint_id", "size"),
                                 fraud=("is_fraud", "sum")).reset_index()
    g = g[g["complaints"] >= MIN_COMPLAINTS].copy()
    g["fraud_rate"] = g["fraud"] / g["complaints"]
    g["rate_rank"] = g["fraud_rate"].rank(method="min", ascending=False).astype(int)
    g["wilson_low"], g["wilson_high"] = wilson_cols(g["fraud"], g["complaints"])
    return g


def q2b(f, top=5):
    g = f.groupby(["product", "company"]).agg(complaints=("complaint_id", "size"),
                                              fraud=("is_fraud", "sum")).reset_index()
    g = g[g["complaints"] >= MIN_COMPLAINTS].copy()
    g["fraud_rate"] = g["fraud"] / g["complaints"]
    g["rate_rank"] = g.groupby("product")["fraud_rate"].rank(method="min",
                                                              ascending=False).astype(int)
    return g[g["rate_rank"] <= top]


def q3(f):
    sizes = f.groupby("company").size()
    companies = sorted(sizes[sizes >= MIN_COMPLAINTS].index)
    counts = f[f["is_fraud"]].groupby(["company", "month"]).size()
    grid = pd.MultiIndex.from_product([companies, MONTHS], names=["company", "month"])
    g = counts.reindex(grid, fill_value=0).rename("fraud").reset_index()
    prev = g.groupby("company")["fraud"].shift(1)
    roll = prev.groupby(g["company"]).rolling(3, min_periods=3)
    g["mean_prev3"] = roll.mean().reset_index(level=0, drop=True)
    g["sd_prev3"] = roll.std(ddof=1).reset_index(level=0, drop=True)
    g = g[g["mean_prev3"].notna()].copy()
    g["sd_used"] = g[["sd_prev3"]].assign(r=g["mean_prev3"] ** 0.5).max(axis=1)
    g["threshold"] = g["mean_prev3"] + 2 * g["sd_used"]
    g["is_spike"] = (g["fraud"] >= SPIKE_MIN) & (g["fraud"] > g["threshold"])
    g["excess"] = g["fraud"] - g["mean_prev3"]
    g["is_biggest"] = False
    cand = g[g["is_spike"] & (g["month"] <= NARRATIVE_LAST_MONTH)]
    if len(cand):
        best = cand.sort_values(["excess", "month", "company"],
                                ascending=[False, True, True]).index[0]
        g.loc[best, "is_biggest"] = True
    return g.drop(columns="sd_prev3")


def q4(f):
    d = f[f["date_received"] <= RESPONSE_TO].copy()
    resp = d["company_response"]
    final = resp.notna() & (resp != "In progress")
    d["final_responses"] = final
    d["timely"] = final & (d["timely_response"] == "Yes")
    d["closed"] = resp.fillna("").str.startswith("Closed")
    d["relief"] = resp.isin(["Closed with monetary relief", "Closed with non-monetary relief"])
    d["monetary_relief"] = resp == "Closed with monetary relief"
    g = d.groupby(["product", "is_fraud"]).agg(
        complaints=("complaint_id", "size"), final_responses=("final_responses", "sum"),
        timely=("timely", "sum"), closed=("closed", "sum"), relief=("relief", "sum"),
        monetary_relief=("monetary_relief", "sum")).reset_index()
    g["timely_rate"] = g["timely"] / g["final_responses"]
    g["relief_rate"] = g["relief"] / g["closed"]
    g["monetary_relief_rate"] = g["monetary_relief"] / g["closed"]
    return g


def q6(f):
    """Follow-up written after Q3: Discover-named companies and Capital One by month."""
    d = f[f["company"].str.upper().str.contains("DISCOVER")
          | (f["company"] == "CAPITAL ONE FINANCIAL CORPORATION")]
    return d.groupby(["month", "company", "product"]).agg(
        complaints=("complaint_id", "size"), fraud=("is_fraud", "sum")).reset_index()


def _norm(v):
    """A CSV cell or a pandas value as text for exact comparison."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return ""
    if isinstance(v, (bool,)) or str(v).lower() in ("true", "false"):
        return str(v).lower()
    try:
        x = float(v)
        return str(int(x)) if x == int(x) else repr(x)
    except (TypeError, ValueError):
        return str(v)


def compare(name, mine, path):
    """Differences between my frame and the saved BigQuery CSV, as lines."""
    _, keys, floats = RESULTS[name]
    theirs = pd.read_csv(path, dtype=str, keep_default_na=False)
    diffs = []
    missing = [c for c in theirs.columns if c not in mine.columns]
    if missing:
        return [f"columns not recomputed: {missing}"]
    if len(theirs) != len(mine):
        diffs.append(f"{len(theirs)} rows in the file, {len(mine)} recomputed")
    mine = mine.copy()
    for k in keys:
        mine[k] = mine[k].map(_norm)
        theirs[k] = theirs[k].map(_norm)
    t = theirs.set_index(keys) if keys else theirs
    m = mine.set_index(keys) if keys else mine
    for idx in t.index:
        if idx not in m.index:
            diffs.append(f"{idx}: in the file, not recomputed")
            continue
        for col in t.columns:
            a, b = t.at[idx, col], m.at[idx, col]
            if col in floats:
                tol = 0.5 * 10 ** -floats[col] + 1e-9
                if (a == "") != pd.isna(b) or (a != "" and abs(float(a) - float(b)) > tol):
                    diffs.append(f"{idx} {col}: file {a}, recomputed {b}")
            elif _norm(a) != _norm(b):
                diffs.append(f"{idx} {col}: file {a}, recomputed {b}")
    return diffs


def recompute(db_path=DB_PATH, results_dir=RESULTS_DIR, mapping_path=mapping.MAPPING_PATH,
              out=print):
    fmap = mapping.load(mapping_path)
    df = load_complaints(db_path)
    f = flag(df, fmap)
    frames = {"q0": q0(f, len(df)), "q1": q1(f), "q2": q2(f), "q2b": q2b(f), "q3": q3(f),
              "q4": q4(f), "q6": q6(f)}
    bad = 0
    for name, (fname, _, _) in RESULTS.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            out(f"{name}: MISSING {path}")
            bad += 1
            continue
        diffs = compare(name, frames[name], path)
        if diffs:
            bad += 1
            out(f"{name}: {len(diffs)} DIFFERENCES")
            for d in diffs[:10]:
                out(f"  {d}")
        else:
            out(f"{name}: matches ({len(frames[name])} rows)")
    return bad, frames


def main(argv):
    try:
        bad, _ = recompute()
    except (RecomputeError, mapping.MappingError, OSError, sqlite3.Error) as e:
        print(f"ERROR {e}")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
