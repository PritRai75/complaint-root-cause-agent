-- Q6 (follow-up, written after Q3's result; not one of the questions fixed
-- before the queries). Discover Bank's fraud complaints fall to 0 from April
-- 2026 while Capital One's rise, and Capital One's May is the biggest spike.
-- Did Discover's complaints stop, or start being filed under Capital One
-- (Capital One completed its acquisition of Discover in 2025)? All complaints
-- and fraud complaints by month for any company named like Discover, and for
-- Capital One. Save as results/q6_discover_followup.csv.
SELECT
  month,
  company,
  product,
  COUNT(*) AS complaints,
  COUNTIF(is_fraud) AS fraud
FROM complaints_analytics.complaints_flagged
WHERE UPPER(company) LIKE '%DISCOVER%'
   OR company = 'CAPITAL ONE FINANCIAL CORPORATION'
GROUP BY month, company, product
ORDER BY company, product, month;
