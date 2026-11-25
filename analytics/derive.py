"""Numbers derived from the saved results, for findings.md (decision 57).

Usage: python -m analytics.derive

Reads only committed files: analytics/results/, analytics/q5_tally.csv,
analytics/fraud_mapping.csv and analytics/issue_values.csv (no database, no BigQuery) and writes analytics/results/derived.csv: one row per
named number, with the file it comes from. Arithmetic only: totals, shares,
differences, Wilson 95% intervals, and the spike rule of 04_spikes.sql applied
again to Capital One and Discover Bank counted as one company (from Q6).
Every number in findings.md must be in a result file or in derived.csv; the
test tests/test_analytics_findings.py checks that.
"""
import csv
import math
import os
import statistics
import sys

from eval import tagrun

RESULTS_DIR = "analytics/results"
TALLY_PATH = "analytics/q5_tally.csv"
MAPPING_PATH = "analytics/fraud_mapping.csv"
VALUES_PATH = "analytics/issue_values.csv"
OUT_NAME = "derived.csv"
CAPITAL_ONE = "CAPITAL ONE FINANCIAL CORPORATION"
DISCOVER = "DISCOVER BANK"
MONTHS = [f"2026-{m:02d}-01" for m in range(1, 9)]
SPIKE_MIN = 20


def read(name, results_dir=RESULTS_DIR):
    path = name if os.sep in name or "/" in name else os.path.join(results_dir, name)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def spike_rule(series):
    """04_spikes.sql's rule on one monthly series (Jan..Aug), April to August."""
    out = []
    for i in range(3, len(series)):
        prev = series[i - 3:i]
        mean = statistics.mean(prev)
        sd = max(statistics.stdev(prev), math.sqrt(mean))
        threshold = mean + 2 * sd
        out.append((MONTHS[i], series[i], mean, sd, threshold,
                    series[i] >= SPIKE_MIN and series[i] > threshold))
    return out


def derive(results_dir=RESULTS_DIR, tally_path=TALLY_PATH, mapping_path=MAPPING_PATH,
           values_path=VALUES_PATH):
    rows = []

    def add(name, value, source):
        rows.append((name, value, source))

    fmap = read(mapping_path)
    key = lambda r: (r["product"], r["issue"], r["sub_issue"])   # noqa: E731
    by_key = {key(r): r for r in fmap}
    add("mapping_rows", len(fmap), "fraud_mapping.csv")
    add("mapping_fraud_rows", sum(r["fraud"] == "yes" for r in fmap), "fraud_mapping.csv")
    add("borderline_left_out_complaints",
        sum(int(v["complaints"]) for v in read(values_path)
            if by_key[key(v)]["borderline"] == "yes" and by_key[key(v)]["fraud"] == "no"),
        "fraud_mapping.csv, issue_values.csv")

    q0 = read("q0_check.csv", results_dir)[0]
    add("fraud_share_all", int(q0["fraud"]) / int(q0["complaints"]), "q0_check.csv")

    q1 = read("q1_monthly.csv", results_dir)
    shares = []
    for m in MONTHS:
        c = sum(int(r["complaints"]) for r in q1 if r["month"] == m)
        f = sum(int(r["fraud"]) for r in q1 if r["month"] == m)
        add(f"month_{m[:7]}_complaints", c, "q1_monthly.csv")
        add(f"month_{m[:7]}_fraud", f, "q1_monthly.csv")
        add(f"month_{m[:7]}_fraud_share", f / c, "q1_monthly.csv")
        shares.append(f / c)
    add("monthly_share_min", min(shares), "q1_monthly.csv")
    add("monthly_share_max", max(shares), "q1_monthly.csv")
    for p in sorted({r["product"] for r in q1}):
        s = [float(r["fraud_share"]) for r in q1 if r["product"] == p]
        tag = "card" if p == "Credit card" else "checking"
        add(f"{tag}_monthly_share_min", min(s), "q1_monthly.csv")
        add(f"{tag}_monthly_share_max", max(s), "q1_monthly.csv")

    q3 = read("q3_spikes.csv", results_dir)
    spikes = [r for r in q3 if r["is_spike"] == "true"]
    add("spike_tests", len(q3), "q3_spikes.csv")
    add("spikes", len(spikes), "q3_spikes.csv")
    tail = 0.5 * math.erfc(2 / math.sqrt(2))        # P(Z > 2) for a normal variable
    add("spikes_expected_by_chance_normal", len(q3) * tail, "q3_spikes.csv")

    q4 = read("q4_response.csv", results_dir)
    for r in q4:
        tag = ("card" if r["product"] == "Credit card" else "checking") + (
            "_fraud" if r["is_fraud"] == "true" else "_other")
        nm = int(r["relief"]) - int(r["monetary_relief"])
        add(f"{tag}_nonmonetary_relief", nm, "q4_response.csv")
        add(f"{tag}_nonmonetary_relief_rate", nm / int(r["closed"]), "q4_response.csv")
        for k, num, den in (("timely", "timely", "final_responses"),
                            ("relief", "relief", "closed"),
                            ("monetary_relief", "monetary_relief", "closed")):
            lo, hi = tagrun.wilson(int(r[num]), int(r[den]))
            add(f"{tag}_{k}_wilson_low", lo, "q4_response.csv")
            add(f"{tag}_{k}_wilson_high", hi, "q4_response.csv")
    for prod in ("card", "checking"):
        for k in ("timely", "relief", "monetary_relief"):
            p = "Credit card" if prod == "card" else "Checking or savings account"
            fr = next(float(r[f"{k}_rate"]) for r in q4 if r["product"] == p and r["is_fraud"] == "true")
            ot = next(float(r[f"{k}_rate"]) for r in q4 if r["product"] == p and r["is_fraud"] == "false")
            add(f"{prod}_{k}_rate_gap_points", 100 * (fr - ot), "q4_response.csv")

    q6 = read("q6_discover_followup.csv", results_dir)

    def series(company, col):
        return [sum(int(r[col]) for r in q6 if r["month"] == m and r["company"] == company)
                for m in MONTHS]

    for col in ("complaints", "fraud"):
        co, di = series(CAPITAL_ONE, col), series(DISCOVER, col)
        both = [a + b for a, b in zip(co, di)]
        for m, a, b, t in zip(MONTHS, co, di, both):
            add(f"q6_{m[:7]}_capital_one_{col}", a, "q6_discover_followup.csv")
            add(f"q6_{m[:7]}_discover_{col}", b, "q6_discover_followup.csv")
            add(f"q6_{m[:7]}_combined_{col}", t, "q6_discover_followup.csv")
        add(f"q6_combined_{col}_mean_jan_mar", statistics.mean(both[:3]), "q6_discover_followup.csv")
        add(f"q6_combined_{col}_mean_apr_aug", statistics.mean(both[3:]), "q6_discover_followup.csv")
        add(f"q6_capital_one_{col}_mean_jan_mar", statistics.mean(co[:3]), "q6_discover_followup.csv")
        add(f"q6_capital_one_{col}_mean_apr_aug", statistics.mean(co[3:]), "q6_discover_followup.csv")
        add(f"q6_discover_{col}_mean_jan_mar", statistics.mean(di[:3]), "q6_discover_followup.csv")
        add(f"q6_discover_{col}_months_after_march", sum(1 for v in di[3:] if v), "q6_discover_followup.csv")
        if col == "fraud":
            judged = spike_rule(both)
            for m, v, mean, sd, th, sp in judged:
                add(f"q6_combined_{m[:7]}_mean_prev3", mean, "q6_discover_followup.csv")
                add(f"q6_combined_{m[:7]}_sd_used", sd, "q6_discover_followup.csv")
                add(f"q6_combined_{m[:7]}_threshold", th, "q6_discover_followup.csv")
            add("q6_combined_spikes", sum(1 for j in judged if j[5]), "q6_discover_followup.csv")

    tally = read(tally_path, results_dir)
    counts = {}
    for r in tally:
        counts[r["read_as"]] = counts.get(r["read_as"], 0) + 1
    for k, v in sorted(counts.items()):
        add(f"q5_read_{k}", v, "q5_tally.csv")
    fraud = len(tally) - counts.get("not_fraud", 0)
    add("q5_read_fraud_any", fraud, "q5_tally.csv")
    add("q5_read_total", len(tally), "q5_tally.csv")
    lo, hi = tagrun.wilson(fraud, len(tally))
    add("q5_fraud_share", fraud / len(tally), "q5_tally.csv")
    add("q5_fraud_share_wilson_low", lo, "q5_tally.csv")
    add("q5_fraud_share_wilson_high", hi, "q5_tally.csv")
    return rows


def write(rows, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, OUT_NAME)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "value", "source"])
        for name, v, src in rows:
            w.writerow([name, v if isinstance(v, int) else f"{v:.6f}", src])
    return path


def main(argv):
    try:
        rows = derive()
    except (OSError, KeyError, ValueError, StopIteration) as e:
        print(f"ERROR {e}")
        return 1
    path = write(rows)
    print(f"{len(rows)} derived numbers")
    print(f"Written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
