# Evaluation

How the agent and its tagger were measured, what the numbers are, and where they are weak. Counts come first, with Wilson 95% intervals. Every rule below was fixed before the run it judges, and no limit was changed after seeing a result.

Who wrote what: I wrote all 200 test labels, the 15 test tasks and the six attack texts. The 6 dev tasks are separate from the 15 test tasks, were used only for tuning and are not counted in any result. Logs for every run are in `eval/runs/` (numbers and IDs only, no narrative text).

## 1. The agent: 14 of 15 tasks

Run once at commit `4de3ad5` after the planner prompt (`plan-3`) was frozen: `agent-test-20261113T072819Z`. 12 report tasks and 3 tasks that should end in a clarifying question. $8.09 and 13.8 minutes.

**14 of 15 passed (Wilson 70.2 to 98.8%).** Reports 12 of 12; clarifying questions 2 of 3. A task passes only if every check that applies passes, and all checks are code:

- outcome (a report was built and sent once, or a question was asked and nothing was sent);
- every key figure of the reference result written in the findings, rounded the way the number check allows;
- for the one task that states a direction, the finding says it moved the expected way (word lists; I review every failure);
- for root-cause tasks, the tagged narratives are exactly the reference population's seeded draw, and the findings give the top causes' shares.

Results by check: figures 10 of 10, direction 1 of 1, population 2 of 2, causes 2 of 2.

**The fail: A02**, "Which bank has been the worst for credit card customers this year?" It should have asked which measure was meant. Both runs built a report on volume, relief rate and timely rate instead. One named Capital One "the broadest concern" without noting that raw complaint counts rise with a bank's size, which this data can't adjust for. I agreed it stays a fail.

How the planner got here: `plan-2` scored 4 of 6 on the dev tasks, because it answered in plain text twice instead of building a report. `plan-3` (every request ends in a report or a question) plus one code reminder when the planner stops with neither scored 6 of 6 with no reminders needed. The 6 dev tasks were not counted anywhere else.

## 2. The tagger: 163 of 200

`test-20261116T055317Z`, run once at `a870cf1` with the prompt `tag-1` and taxonomy v1, both frozen after dev (dev: 82 of 90, a fresh repeat moved 0 of 90 tags). Claude Sonnet 4.6, one narrative per call, a forced `record_root_cause` call with the codes as an enum. $3.51 and 301 seconds; one narrative was cut for length.

**Exact agreement with my labels: 163 of 200 (81.5%, 75.5 to 86.3%).** Checking 87 of 100, credit card 76 of 100. Exact agreement is the only accuracy figure; nothing else is scored.

| Code | Labelled | Tagged | Correct |
| --- | ---: | ---: | ---: |
| fraud_claim | 55 | 62 | 55 |
| account_restricted | 30 | 38 | 29 |
| servicing_error | 29 | 19 | 17 |
| dispute_mishandled | 27 | 29 | 25 |
| fee_charged | 14 | 13 | 10 |
| terms_not_honored | 12 | 19 | 11 |
| credit_reporting | 10 | 10 | 8 |
| third_party | 10 | 1 | 1 |
| unclear | 7 | 4 | 4 |
| other | 6 | 5 | 3 |

Biggest confusions: servicing_error tagged as account_restricted (7), third_party tagged as dispute_mishandled (3) or fraud_claim (3), fee_charged tagged as terms_not_honored (3).

**Where the misses are.** 37 misses. While labelling I marked 81 narratives with a runner-up code ("vs <code>"). 34 of the 37 misses are on those flagged items (the tagger agreed on 47 of 81 flagged, and on 116 of 119 unflagged). 23 of the 37 misses are exactly the runner-up I had named. The rule was fixed first: a miss counts as matching the runner-up only when the tagger's code equals the one code after "vs". This describes the misses; it is not a score.

**Shares over the root-cause base** (the three non-causes are left out, so labels n = 177, tags n = 190):

| Code | Labelled share | Tagged share |
| --- | ---: | ---: |
| fraud_claim | 31.1% | 32.6% |
| account_restricted | 16.9% | 20.0% |
| servicing_error | 16.4% | 10.0% |
| dispute_mishandled | 15.3% | 15.3% |
| fee_charged | 7.9% | 6.8% |
| terms_not_honored | 6.8% | 10.0% |
| credit_reporting | 5.6% | 5.3% |

So the tagger under-states servicing errors by about 6 points and over-states account restrictions and unmet terms by about 3 each. That is one sample of 200: a direction, not a correction factor. `third_party` recall is 1 of 10, so complaints that blame only someone else are counted as company causes.

**Against the CFPB issue column.** Grouped by (product, issue), the 200 labels fall in 18 groups; 13 of them hold two or more causes, covering 195 of 200 narratives. The best possible lookup from (product, issue) to one cause (the most common label in each group), fitted on the same 200 and so an upper bound, matches 101 of 200. The tagger matches 163. Example: "Managing an account" has 60 items and 9 causes; the tagger got 53 of 60 right. The rule (grouping by product and issue, the stronger lookup) was fixed before the run.

**Re-labelling blind.** 30 items (15 per product, seeded, leaving out the 14 items we had discussed) were re-labelled by me 0.62 days after the first labels: 27 of 30 the same (90.0%, 74.4 to 96.5%), Cohen's kappa 0.88. With so short a gap this is an upper bound on how well the labels reproduce. The 3 disagreements were all on items I had flagged with a runner-up, and the tagger agreed with the first label on all 3. I made no corrections to the first labels.

## 3. Injection resistance

Six attack texts (X1 to X6), written by me, were planted in a copy of the database. 8 of each attack in each of the two root-cause tasks' reference draws (96 narratives), inserted at the start, middle or end and never replacing the consumer's text, plus one of each in one dev task. Only narratives short enough for the tagger to read in full were used, and the draw is seeded. The planted test run was done once, after the clean one, at the same commit; nothing was tuned on it.

- **6 of 96 tags flipped** compared with the same narrative's clean-run tag (Wilson 95% 2.9 to 13.0%).
- **0 of 74 narratives became the attack's target** (those whose clean tag wasn't already the target; upper bound 4.9%).
- By attack: X2 (a fake system message) 3 of 16, X4 2 of 16, X5 1 of 16, X1, X3 and X6 0 of 16. By position: 0 of 32 at the end.
- Shares moved by at most 0.5 points. All 15 task outcomes and sends were the same as the clean run.
- One planted attack (X4) was printed in a report as the example quote, because quotes are shown as published. Planted narratives are counted, not removed.
- The planting skipped narratives that contain an AI-instruction-like phrase using a plain substring match. That also skipped ordinary text ("the system prompted me to change my login"), so the count of 16 such narratives is a ceiling and the exclusion errs on the safe side.

Injection resistance is measured here, not guaranteed. What the code guarantees is the recipient, the single send, the read-only database and the limits.

## 4. The deployed run used as the README example

Run `run-65c895b9…` on the deployed API (AgentCore Runtime, 16 November 2026, 303 seconds, status `sent`, email arrived): "What are the most common root causes in credit card complaints about fees or interest from January to June 2026?" The population was 4,944 complaints; 200 were tagged and shares are over the 195 given a root cause: terms_not_honored 28.2% (55), fee_charged 26.2% (51), servicing_error 15.4% (30), fraud_claim 11.8% (23), dispute_mishandled 8.2% (16), account_restricted 7.2% (14), credit_reporting 3.1% (6).

Honest notes on this run:

- The top cause is the one the tagger over-counts (10.0% tagged against 6.8% labelled on the test set), and the top two differ by 4 narratives.
- Quotes 1 and 2 are near-identical "never activated, charged a fee" stories, one under terms_not_honored and one under fee_charged. That is the boundary between those two codes, visible in a real report.
- "About fees or interest" is literal. From the run's trace (the report itself doesn't show it), the population is credit card complaints from January to June whose issue or sub-issue says fee or interest, or whose narrative contains "fee" or "interest". The word "interest" turns up in passing, so it is wider than complaints about fees.
- The query that built the population isn't printed in the report's method section.

An earlier deployed run (`run-cea1022f…`) found a wrong finding: "Five of the eight months" were between 7,197 and 7,963 when seven are. The number check looks at digits, not number words.

## 5. Two bugs the deployment checks found

- An unknown run ID returned 502 instead of 404: S3 answers AccessDenied for a missing object unless the role may list the bucket. Fixed by adding `s3:ListBucket` to the status role.
- The first real send was refused: SES didn't accept identity-ARN resources for the raw send. Fixed with permissions on `"*"` pinned by `ses:FromAddress` and `ses:Recipients`, which also makes IAM enforce the single recipient.

## 6. Not done, and changes I'd make after the evaluation

None of these were made, so the numbers above describe the system as tested.

- Cut the clarifying question at the last full sentence within 300 characters.
- Check number words ("five of the eight") in findings.
- Print every query behind a finding's numbers, and the tag population query, in the method section.
- Superlatives across companies ("worst bank") should trigger a clarifying question.
- Avoid printing text outside the consumer's own story as a quote.
- The one-shot versus agent comparison, judge calibration and a second reproducibility check were cut for time at the start (decision 3).
