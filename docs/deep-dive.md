# Fraud and scam claims that issue-based routing misses

A deep-dive memo written as practice, on public CFPB data: credit card and checking complaints received from January to August 2026. No bank has used it.

**Recommendation.** Run a human-reviewed pilot that tags fraud or scam claims in the narratives of complaints filed under non-fraud issues, and routes the tagged ones to the fraud team. On the held-out test set the tagger found every such claim, and about 1,085 narratives a month look like this (730.6 to 1,568.0).

## The problem

Complaints are routed by the issue the consumer picks when filing. Five of the 103 issue and sub-issue categories describe fraud (unauthorized transactions and accounts opened without consent), and they hold 18,190 of the 125,220 complaints (14.5%), between 2,095 and 2,588 a month. Those complaints reach a fraud team by their category alone.

The narrative tells a different story for many complaints. In the 200 held-out test narratives, 36 of the 169 filed under non-fraud issues describe a fraud or scam claim the company mishandled: a transaction the consumer says someone else made, or one they were tricked into, which the company denied, delayed or didn't investigate. 27 of the 36 are checking complaints. Routed by issue, none of them reaches the fraud team by its category. The category is noisy in the other direction too: of the 31 test narratives filed under fraud issues, 19 are fraud or scam claims mishandled, and the rest have other root causes.

## How big it is

The share comes from the test set, which was drawn at random within each month, 100 narratives per product, one per distinct text. The base is the same pool of distinct narratives (35,627 under non-fraud issues, January to July). The estimate a month is the share times the base, over 7 months.

| Product | Fraud or scam claims under non-fraud issues (test set) | Share (95% interval) | Estimate a month (interval) |
| --- | --- | --- | --- |
| Checking or savings account | 27 of 86 | 31.4% (22.6 to 41.8%) | 815.0 (585.7 to 1,085.7) |
| Credit card | 9 of 83 | 10.8% (5.8 to 19.3%) | 270.4 (144.9 to 482.3) |
| Both | 36 of 169 | | about 1,085 (730.6 to 1,568.0) |

The range for both products adds the two products' bounds, so it is wider than a joint interval. Two assumptions sit under it, and the memo states them rather than tests them:

- **The test-set share holds for every narrative**, copies included. The draw used one narrative per distinct text.
- **The figure counts published narratives only.** About a third of complaints have one, and consumers choose whether to publish, so it can't be scaled up to all complaints. In a bank the complaint text is in-house, and the real figure would be measured there.

Reviewer time saved would need a minutes-per-complaint figure. None is measured here, so any time saving in a pilot plan is an assumption to replace with the pilot's own timing.

## Can the tagger find them?

The tagger reads one narrative at a time and gives it one of 10 root-cause codes; fraud_claim is the code for a fraud or scam claim mishandled.

- **Overall:** it matches the hand label on 163 of 200 test narratives (81.5%, 75.5 to 86.3%). The best possible lookup from the issue column, fitted on the same 200, matches 101.
- **On fraud claims:** recall is 55 of 55 (93.5 to 100%) and precision 55 of 62 (78.5 to 94.4%).
- **On exactly this group,** the narratives under non-fraud issues: it tagged 40 as fraud_claim, 36 correctly, and found all 36 labelled fraud claims.
- **Its main error:** 3 of the 7 extra fraud tags were narratives labelled third_party, where the consumer blames someone other than the company. Routed to the fraud team, those cost review time but have no company failure to fix.

## Success metrics for a pilot

Set every threshold before the pilot starts.

| Metric | How it is measured | Role |
| --- | --- | --- |
| Fraud or scam claims found a month outside fraud issues | Tags confirmed by the reviewer | Primary |
| Precision | Confirmed tags over all fraud_claim tags | Primary |
| Recall | A fresh hand-labelled sample of 100 narratives under non-fraud issues, tagged blind | Primary |
| False-route rate | Tagged claims the fraud team sends back | Guardrail |
| Reviewer minutes per tag | Timed in the pilot | Cost |
| Monthly fraud share, with a rule-based spike check | Trailing 3-month mean plus 2 standard deviations, per company | Monitoring |

## Risks

- **A small test set.** 55 fraud claims in total and 36 in this group. The checking interval runs from 22.6 to 41.8%. The fresh sample of 100 is the first thing the pilot should do.
- **Two definitions.** The issue mapping counts unauthorized transactions and new accounts, but not scams. fraud_claim includes scams the company mishandled. The pilot needs one agreed definition for the fraud team.
- **Spikes that are filing changes.** The largest spike the monthly check flagged, Capital One in May (187 against a threshold of 181.4), came from Discover Bank's complaints moving under Capital One. Counted together, no month is flagged. Any monitoring has to check company-level changes before calling a spike new fraud.
- **Published text.** CFPB narratives stop in mid-August 2026, so this public version can't be refreshed; a bank's own complaint text doesn't have that limit.

## Where the figures come from

- Sizing: [analytics/routing_size.py](../analytics/routing_size.py) writes [analytics/results/q7_routing_size.csv](../analytics/results/q7_routing_size.csv). It rebuilds the test set's pool and stops if the pool differs from the one the test set was drawn from.
- Tagger accuracy: [docs/evaluation.md](evaluation.md).
- Fraud complaints, spikes and Discover Bank: [analytics/findings.md](../analytics/findings.md).

Every number in this memo appears in a result file or in `docs/evaluation.md`; `tests/test_analytics_findings.py` checks it.
