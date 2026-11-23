-- Q3. Where were the spikes? (decision 56)
-- Companies with at least 200 complaints, January to August 2026, both
-- products. Every company gets all 8 months, zero-filled. A month is a spike
-- when its fraud count is at least 20 and above
--   mean of the 3 months before it + 2 * max(sample SD of those 3, sqrt(mean)).
-- Only April to August have 3 months before them. excess = fraud - mean_prev3.
-- The biggest spike for question 5 is the spike with the largest excess in
-- April to July (narratives stop in July; ties: earlier month, then company).
-- Save as results/q3_spikes.csv (every company-month April to August).
WITH companies AS (
  SELECT company
  FROM complaints_analytics.complaints_flagged
  GROUP BY company
  HAVING COUNT(*) >= 200
),
months AS (
  SELECT month
  FROM UNNEST(GENERATE_DATE_ARRAY(DATE '2026-01-01', DATE '2026-08-01', INTERVAL 1 MONTH)) AS month
),
counts AS (
  SELECT company, month, COUNTIF(is_fraud) AS fraud
  FROM complaints_analytics.complaints_flagged
  GROUP BY company, month
),
grid AS (
  SELECT c.company, m.month, IFNULL(n.fraud, 0) AS fraud
  FROM companies AS c
  CROSS JOIN months AS m
  LEFT JOIN counts AS n USING (company, month)
),
windowed AS (
  SELECT
    company,
    month,
    fraud,
    AVG(fraud) OVER prev3 AS mean_prev3,
    STDDEV_SAMP(fraud) OVER prev3 AS sd_prev3,
    COUNT(*) OVER prev3 AS n_prev
  FROM grid
  WINDOW prev3 AS (PARTITION BY company ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
),
judged AS (
  SELECT
    *,
    GREATEST(IFNULL(sd_prev3, 0), SQRT(mean_prev3)) AS sd_used,
    mean_prev3 + 2 * GREATEST(IFNULL(sd_prev3, 0), SQRT(mean_prev3)) AS threshold
  FROM windowed
  WHERE n_prev = 3
)
SELECT
  company,
  month,
  fraud,
  ROUND(mean_prev3, 3) AS mean_prev3,
  ROUND(sd_used, 3) AS sd_used,
  ROUND(threshold, 3) AS threshold,
  fraud >= 20 AND fraud > threshold AS is_spike,
  ROUND(fraud - mean_prev3, 3) AS excess,
  (fraud >= 20 AND fraud > threshold AND month <= DATE '2026-07-01')
    AND ROW_NUMBER() OVER (
      PARTITION BY (fraud >= 20 AND fraud > threshold AND month <= DATE '2026-07-01')
      ORDER BY fraud - mean_prev3 DESC, month, company) = 1 AS is_biggest
FROM judged
ORDER BY company, month;
