-- Q2. Which companies have the highest fraud-complaint rate (fraud complaints
-- / all complaints), counting only companies with at least 200 complaints?
-- Both products together, January to August 2026. Wilson 95% interval on the
-- rate. Ties share a rank. Save as results/q2_company_rates.csv.
WITH by_company AS (
  SELECT
    company,
    COUNT(*) AS complaints,
    COUNTIF(is_fraud) AS fraud
  FROM complaints_analytics.complaints_flagged
  GROUP BY company
  HAVING COUNT(*) >= 200
),
rated AS (
  SELECT
    company,
    complaints,
    fraud,
    SAFE_DIVIDE(fraud, complaints) AS p,
    1.96 AS z
  FROM by_company
)
SELECT
  RANK() OVER (ORDER BY SAFE_DIVIDE(fraud, complaints) DESC) AS rate_rank,
  company,
  complaints,
  fraud,
  ROUND(p, 4) AS fraud_rate,
  ROUND(GREATEST(0, (p + z * z / (2 * complaints)
        - z * SQRT(p * (1 - p) / complaints + z * z / (4 * complaints * complaints)))
        / (1 + z * z / complaints)), 4) AS wilson_low,
  ROUND(LEAST(1, (p + z * z / (2 * complaints)
        + z * SQRT(p * (1 - p) / complaints + z * z / (4 * complaints * complaints)))
        / (1 + z * z / complaints)), 4) AS wilson_high
FROM rated
ORDER BY rate_rank, company;
