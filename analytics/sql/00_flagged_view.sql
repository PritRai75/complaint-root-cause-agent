-- 00. The fraud flag on every complaint (decisions 55, 59).
-- Joins each complaint to its row of the fraud mapping. A blank sub_issue is
-- NULL in both tables, so the join compares IFNULL(sub_issue, '').
-- Run once; the questions and the Looker Studio dashboard read this view.
CREATE OR REPLACE VIEW complaints_analytics.complaints_flagged AS
SELECT
  c.complaint_id,
  c.date_received,
  DATE_TRUNC(c.date_received, MONTH) AS month,
  c.product,
  c.issue,
  c.sub_issue,
  c.company,
  c.company_response,
  c.timely_response,
  c.has_narrative,
  m.fraud = 'yes' AS is_fraud,
  m.kind AS fraud_kind
FROM complaints_analytics.complaints AS c
JOIN complaints_analytics.fraud_mapping AS m
  ON c.product = m.product
 AND c.issue = m.issue
 AND IFNULL(c.sub_issue, '') = IFNULL(m.sub_issue, '');
