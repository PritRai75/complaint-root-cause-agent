-- Q4. Do companies respond differently to fraud complaints? Complaints
-- received January to June 2026 only (decision 7). Timely response over final
-- responses (company_response not 'In progress'); relief and monetary relief
-- over closed complaints (decision 14, the agent's own definitions).
-- Save as results/q4_response.csv.
SELECT
  product,
  is_fraud,
  COUNT(*) AS complaints,
  COUNTIF(company_response <> 'In progress') AS final_responses,
  COUNTIF(company_response <> 'In progress' AND timely_response = 'Yes') AS timely,
  COUNTIF(company_response LIKE 'Closed%') AS closed,
  COUNTIF(company_response IN ('Closed with monetary relief', 'Closed with non-monetary relief')) AS relief,
  COUNTIF(company_response = 'Closed with monetary relief') AS monetary_relief,
  ROUND(SAFE_DIVIDE(COUNTIF(company_response <> 'In progress' AND timely_response = 'Yes'),
                    COUNTIF(company_response <> 'In progress')), 4) AS timely_rate,
  ROUND(SAFE_DIVIDE(COUNTIF(company_response IN ('Closed with monetary relief', 'Closed with non-monetary relief')),
                    COUNTIF(company_response LIKE 'Closed%')), 4) AS relief_rate,
  ROUND(SAFE_DIVIDE(COUNTIF(company_response = 'Closed with monetary relief'),
                    COUNTIF(company_response LIKE 'Closed%')), 4) AS monetary_relief_rate
FROM complaints_analytics.complaints_flagged
WHERE date_received BETWEEN DATE '2026-01-01' AND DATE '2026-06-30'
GROUP BY product, is_fraud
ORDER BY product, is_fraud;
