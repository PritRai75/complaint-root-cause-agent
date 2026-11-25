-- Q1. How did fraud complaints move month by month, in count and as a share
-- of all complaints? By product, January to August 2026 (volumes use all
-- eight months, decision 7). Save as results/q1_monthly.csv.
WITH monthly AS (
  SELECT
    month,
    product,
    COUNT(*) AS complaints,
    COUNTIF(is_fraud) AS fraud,
    COUNTIF(fraud_kind = 'unauthorized_transaction') AS unauthorized_transaction,
    COUNTIF(fraud_kind = 'new_account') AS new_account
  FROM complaints_analytics.complaints_flagged
  GROUP BY month, product
)
SELECT
  month,
  product,
  complaints,
  fraud,
  unauthorized_transaction,
  new_account,
  ROUND(SAFE_DIVIDE(fraud, complaints), 4) AS fraud_share,
  fraud - LAG(fraud) OVER (PARTITION BY product ORDER BY month) AS fraud_change
FROM monthly
ORDER BY product, month;
