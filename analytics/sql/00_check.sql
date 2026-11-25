-- 00. Check the view: every complaint joined to exactly one mapping row.
-- Expect complaints 125220, joined 125220, fraud 18190. Save as results/q0_check.csv.
SELECT
  (SELECT COUNT(*) FROM complaints_analytics.complaints) AS complaints,
  COUNT(*) AS joined,
  COUNT(DISTINCT complaint_id) AS distinct_ids,
  COUNTIF(is_fraud) AS fraud,
  COUNTIF(fraud_kind = 'unauthorized_transaction') AS unauthorized_transaction,
  COUNTIF(fraud_kind = 'new_account') AS new_account
FROM complaints_analytics.complaints_flagged;
