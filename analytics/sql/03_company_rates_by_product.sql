-- Q2b. The 5 highest fraud-complaint rates within each product, among
-- companies with at least 200 complaints in that product. QUALIFY keeps the
-- top 5 per product (ties share a rank, so a group can hold more than 5).
-- WHERE TRUE keeps QUALIFY valid on BigQuery versions that require a WHERE.
-- Save as results/q2b_company_rates_by_product.csv.
WITH by_company AS (
  SELECT
    product,
    company,
    COUNT(*) AS complaints,
    COUNTIF(is_fraud) AS fraud
  FROM complaints_analytics.complaints_flagged
  GROUP BY product, company
  HAVING COUNT(*) >= 200
)
SELECT
  product,
  RANK() OVER (PARTITION BY product ORDER BY SAFE_DIVIDE(fraud, complaints) DESC) AS rate_rank,
  company,
  complaints,
  fraud,
  ROUND(SAFE_DIVIDE(fraud, complaints), 4) AS fraud_rate
FROM by_company
WHERE TRUE
QUALIFY rate_rank <= 5
ORDER BY product, rate_rank, company;
