# Fraud complaint dashboard (Looker Studio)

**Link:** _added once the dashboard is built and shared (view only)_

One page over the BigQuery sandbox dataset `complaints_analytics`. The numbers it shows are the same as `findings.md`, because it reads the same view and queries.

## Data sources

| Name in Looker Studio | BigQuery source | Feeds |
| --- | --- | --- |
| Flagged complaints | view `complaints_analytics.complaints_flagged` | scorecards, monthly charts, filters |
| Company rates | custom query: contents of `sql/02_company_rates.sql` | top-10 bar chart |
| Responses | custom query: contents of `sql/05_response.sql` | fraud vs other table |
| Discover follow-up | custom query: contents of `sql/06_discover_followup.sql` | Capital One and Discover line chart |

Calculated fields on **Flagged complaints**:

| Field | Formula |
| --- | --- |
| Fraud complaints | `SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)` |
| Fraud share | `SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(complaint_id)` (format: percent) |

## Layout

| Position | Chart | Setup |
| --- | --- | --- |
| Top row | 3 scorecards | `COUNT(complaint_id)` as "Complaints"; Fraud complaints; Fraud share |
| Top right | 3 filter controls | drop-downs on `product` and `company`; date range on `date_received` |
| Second row, left | Time series | dimension `month`, metric Fraud complaints, breakdown `product` |
| Second row, right | Time series | dimension `month`, metric Fraud share, breakdown `product` |
| Third row, left | Bar chart (Company rates) | dimension `company`, metric `fraud_rate` (percent), sorted descending, 10 bars |
| Third row, right | Table (Responses) | `product`, `is_fraud`, `timely_rate`, `relief_rate`, `monetary_relief_rate` (percent) |
| Bottom | Time series (Discover follow-up) | dimension `month`, metric `complaints`, breakdown `company`, filtered to Credit card |

Text boxes:

- Under the company bar chart: "Companies with at least 200 complaints. The credit bureaus top the list because identity theft is reported to them."
- Above the bottom chart: "Discover Bank's complaints stop after March 2026 as Capital One's rise: a change in filing, not a fraud spike."
- Footer: "Complaints are not fraud incidence. Responses: complaints received January to June. Source: CFPB complaint database; see findings.md."

The filter controls apply to the Flagged complaints charts only. The three custom-query charts show their whole query.

## Sharing

**Share** → **Manage access** → anyone with the link can **view**. Keep the data sources' credentials set to the owner's (the default), so viewers don't need BigQuery access.

In the sandbox, tables and views expire 60 days after they are created. Before showing the dashboard after that, rebuild the CSV with `python -m analytics.export table`, upload it again as `complaints`, upload `fraud_mapping.csv` again, and run `sql/00_flagged_view.sql` again. The dashboard then works as before.

Screenshot: `img/dashboard.png`.
