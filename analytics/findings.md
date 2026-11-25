# Fraud complaint trends, January to August 2026

Fraud complaints were a steady 14.5% of credit card and checking complaints, month after month. The largest "spike" in the period was Capital One picking up Discover Bank's complaints, not new fraud.

Every number below comes from a file in `results/`, saved from BigQuery or derived from those files by `python -m analytics.derive`. Every BigQuery result was also recomputed independently in pandas from the source database, and all of them match (`python -m analytics.recompute`). A test checks that each number in this file appears in one of those files.

## Data and definition

- **Data:** 125,220 CFPB complaints about credit cards and checking or savings accounts, received January to August 2026, loaded into a BigQuery sandbox (`sql/`).
- **Fraud complaint:** a complaint whose issue and sub-issue are one of 5 of the 103 values in the data (`fraud_mapping.csv`, written and committed before any query). That gives 18,190 fraud complaints: 11,220 unauthorized transactions and 6,970 accounts opened without the consumer's consent.
- **Scams are not separated.** No card or checking category isolates a consumer tricked into paying. Scams sit inside the fraud categories and inside broader ones such as purchase disputes and payment problems.
- **Left out:** 1,510 complaints in borderline categories, such as credit report information belonging to someone else, where the category can't tell fraud from other causes.

## 1. The share of fraud complaints held steady

Fraud complaints were between 14.0% and 15.0% of all complaints in every month (both products together). Checking ranged from 13.4% to 15.0% and credit cards from 14.2% to 16.1%, with no trend in either (`results/q1_monthly.csv`).

## 2. The highest rates are where identity theft gets reported

Among the 53 companies with at least 200 complaints, the three credit bureaus have the highest fraud-complaint rates: Experian 51.6% (Wilson 95% interval 48.9 to 54.3), Equifax 49.2% and TransUnion 49.1%. Their fraud complaints are mostly people reporting accounts opened in their name, so the rate shows where identity theft is reported, not fraud at the bureau.

Among the seven companies with the most complaints, Chime's rate is 25.2% (24.1 to 26.3), well above the banks'. The banks run from 10.0% (Citibank) to 15.3% (Wells Fargo) (`results/q2_company_rates.csv`; the top five within each product are in `results/q2b_company_rates_by_product.csv`).

Discover Bank's 13.8% covers January to March only, and Capital One's 11.0% includes Discover's complaints from April (finding 3).

## 3. Of 7 flagged spikes, the biggest was a filing change

A company-month counts as a spike when its fraud complaints are at least 20 and above the mean of the 3 months before it plus 2 standard deviations, with the standard deviation floored at the square root of the mean. The rule covers companies with at least 200 complaints, April to August.

It flagged 7 of 265 company-months (`results/q3_spikes.csv`). If counts varied normally around their trailing mean, about 6 of 265 would cross that line by chance. So a flag here is a prompt to look, not evidence of a fraud wave.

The biggest flag was Capital One in May 2026: 187 fraud complaints against a trailing mean of 155.3 and a threshold of 181.4. A follow-up query, written after seeing this result, explains it (`sql/06_discover_followup.sql`, `results/q6_discover_followup.csv`):

| | January to March (monthly mean) | April to August (monthly mean) |
| --- | ---: | ---: |
| Discover Bank, all complaints | 299.7 | none filed |
| Capital One + Discover Bank, all complaints | 1,689 | 1,649.2 |
| Capital One, fraud complaints | 158 | 178.8 |
| Capital One + Discover Bank, fraud complaints | 199.3 | 178.8 |

Discover Bank's complaints stop after March, and Capital One's rise from April. Counted as one company, both totals are level, and the same spike rule flags no month (in May, 187 against a threshold of 206.8). One of the narratives read for May (finding 5) is a Discover appeal letter filed under Capital One.

This fits a change in which company name complaints are filed under after Capital One's acquisition of Discover. The data shows the switch, not its cause. A spike in one company's line should be checked against company-level changes before anyone investigates it as fraud.

## 4. Card fraud complaints get relief more often, but less often in money

These comparisons use complaints received January to June, because July and August responses aren't final (`results/q4_response.csv`). Timely response is measured over final responses; relief is measured over closed complaints.

- **Checking:** fraud and other complaints are handled almost the same. Timely response is 99.1% vs 99.3%, and relief is 18.0% (17.1 to 19.0) vs 18.5% (18.1 to 18.9).
- **Credit card, relief overall:** fraud complaints get relief more often, 33.6% (32.5 to 34.7) vs 25.1% (24.6 to 25.5), a gap of 8.5 points.
- **Credit card, monetary relief:** fraud complaints get it less often, 13.5% (12.7 to 14.3) vs 16.4% (16.1 to 16.8).
- **Credit card, non-monetary relief:** this is where the difference lies, 20.1% of closed fraud complaints vs 8.6% of others. That fits new-account fraud, where the fix is closing the account or correcting the credit report rather than paying money back.

## 5. What the biggest-spike month's complaints say

25 of the 72 Capital One fraud complaints with a narrative received in May 2026 were drawn at random (seed 20261123) and read (`results/q5_pool.csv`, `results/q5_sample_ids.csv`, `q5_tally.csv`):

| Read as | Count |
| --- | ---: |
| Account opened in the consumer's name (identity theft) | 10 |
| Unauthorized transaction | 6 |
| Scam: consumer tricked into a payment or into sharing access | 2 |
| Not fraud: merchant or subscription disputes, a service paid for and not delivered, a card approved for a different product than shown | 7 |

So 18 of 25 (72%, Wilson 95% interval 52.4 to 85.7) fit the fraud definition. The issue categories over-count fraud by including merchant disputes that the consumer calls unauthorized, and 2 of the 18 are scams the consumer was tricked into. No shared merchant or method stands out. The common thread is a claim denied without the evidence being explained.

## Caveats

- **Complaints are not fraud incidence.** A complaint exists only when a consumer chooses to file one with the CFPB.
- **Larger companies get more complaints,** so companies are compared by rate, not by count. A rate here is over complaints, not over customers or accounts.
- **Narratives are published only with the consumer's consent,** so the reading in finding 5 covers that subset.
- **The fraud definition depends on the consumer's choice of category.** Finding 5 suggests it includes non-fraud disputes, and scams outside these categories are missed.
- **Questions 1 to 5 and their rules were fixed before any query ran;** question 6 was added after question 3's result.

## Reproduce

| Step | Command or file |
| --- | --- |
| Export from the source database | `python -m analytics.export values` and `python -m analytics.export table` |
| Load into BigQuery | `schema.json`, `mapping_schema.json` (console upload, dataset `complaints_analytics`) |
| Run the queries | `sql/00_flagged_view.sql` to `sql/06_discover_followup.sql`, results saved to `results/` |
| Check every result | `python -m analytics.recompute` |
| Draw the reading sample | `python -m analytics.spike_sample` |
| Derived numbers | `python -m analytics.derive` |
