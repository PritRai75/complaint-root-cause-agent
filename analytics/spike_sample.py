"""Draw the narratives to read for question 5 (decision 58).

Usage: python -m analytics.spike_sample

Takes the one company-month marked is_biggest in analytics/results/q3_spikes.csv
(the spike with the largest excess, April to July; narratives stop in July),
finds that company's fraud complaints received that month that carry a
narrative, and draws 25 of them at random with seed 20261123 (all of them if
there are fewer). Writes:

  analytics/results/q5_sample_ids.csv   IDs, product, issue, sub-issue (committed)
  analytics/results/q5_pool.csv         the company-month, pool size, number drawn, seed
  scratch/q5_reading.md                 the narratives to read (gitignored)

How each narrative was read goes in analytics/q5_tally.csv (item, complaint ID,
read_as, note); findings.md reports the counts.
"""
import csv
import os
import random
import sqlite3
import sys

from analytics import mapping

DB_PATH = "data/complaints.sqlite"
SPIKES_PATH = "analytics/results/q3_spikes.csv"
IDS_PATH = "analytics/results/q5_sample_ids.csv"
POOL_PATH = "analytics/results/q5_pool.csv"
READING_PATH = "scratch/q5_reading.md"
SEED = 20261123
N = 25


class SampleError(Exception):
    pass


def biggest(spikes_path=SPIKES_PATH):
    with open(spikes_path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["is_biggest"].lower() == "true"]
    if len(rows) != 1:
        raise SampleError(f"{spikes_path} marks {len(rows)} rows as is_biggest, not 1")
    r = rows[0]
    if r["month"] > "2026-07-01":
        raise SampleError(f"the biggest spike is in {r['month']}, after narratives stop")
    return r["company"], r["month"]


def candidates(company, month, fmap, db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise SampleError(f"{db_path} not found")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT c.complaint_id, c.date_received, c.product, c.issue, "
            "COALESCE(c.sub_issue, ''), n.narrative FROM complaints c "
            "JOIN narratives n USING (complaint_id) "
            "WHERE c.company = ? AND substr(c.date_received, 1, 7) = ? "
            "ORDER BY c.complaint_id", (company, month[:7])).fetchall()
    finally:
        conn.close()
    return [r for r in rows if fmap[(r[2], r[3], r[4])]["fraud"] == "yes"]


def draw(rows, n=N, seed=SEED):
    if len(rows) <= n:
        return list(rows)
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r[0])


def main(argv):
    try:
        fmap = mapping.load()
        company, month = biggest()
        pool = candidates(company, month, fmap)
        picked = draw(pool)
    except (SampleError, mapping.MappingError, OSError, sqlite3.Error) as e:
        print(f"ERROR {e}")
        return 1
    os.makedirs(os.path.dirname(IDS_PATH), exist_ok=True)
    with open(IDS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["complaint_id", "date_received", "product", "issue", "sub_issue"])
        w.writerows(r[:5] for r in picked)
    with open(POOL_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["company", "month", "fraud_with_narrative", "drawn", "seed"])
        w.writerow([company, month, len(pool), len(picked), SEED])
    os.makedirs(os.path.dirname(READING_PATH), exist_ok=True)
    with open(READING_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Question 5 reading: {company}, {month[:7]}\n\n")
        f.write(f"{len(picked)} of {len(pool)} fraud complaints with a narrative, seed {SEED}.\n")
        for i, r in enumerate(picked, 1):
            f.write(f"\n## {i}. {r[0]} ({r[1]}, {r[2]})\n\n{r[3]} / {r[4] or '(no sub-issue)'}"
                    f"\n\n{r[5]}\n")
    print(f"Biggest spike: {company}, {month[:7]}; {len(pool)} fraud complaints with a "
          f"narrative; drew {len(picked)}")
    print(f"Written: {IDS_PATH}")
    print(f"Written: {POOL_PATH}")
    print(f"Written: {READING_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
