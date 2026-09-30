# Data Dictionary — Customer Churn Dataset

**Source file:** `data/raw/Customer_Data.csv`
**Rows:** 6,418 customers | **Columns:** 32

## ⚠️ Churn definition used in this project

This dataset is a **snapshot** of each customer's status at one point in time — it does not
contain purchase-level timestamps, so a literal "will churn in the next 30 days" label cannot
be constructed from it. Instead, `Customer_Status` already encodes the outcome:

| Value | Meaning | Used as |
|---|---|---|
| `Stayed` | Active customer, no churn | **Negative class (0)** |
| `Churned` | Customer has left | **Positive class (1)** |
| `Joined` | Brand-new customer, too early to know outcome | **Excluded from modeling** (no history to learn from, would bias the base rate) |

So the modeling target is: **"Given a customer's profile, usage and billing history, is this
customer a Churned customer or a Stayed customer?"** — a churn-risk classifier trained on the
final observed outcome, exactly like a real company would train on last quarter's churned
customers before applying it to *current* active customers to score forward-looking risk.

## 🚫 Leakage columns — excluded from features

| Column | Why it leaks |
|---|---|
| `Churn_Category` | Only populated **after** a customer has churned — would not exist at prediction time for an active customer. |
| `Churn_Reason` | Same — a post-churn exit-survey field. |
| `Customer_Status` | This **is** the target; encoded separately as `churn` (1/0). |

These are kept in the raw data for reporting ("why did people leave") but never fed to the model.

## Feature reference

| Feature | Meaning |
|---|---|
| `Customer_ID` | Unique customer identifier (dropped before modeling) |
| `Gender` | Male / Female |
| `Age` | Customer age in years |
| `Married` | Marital status (Yes/No) |
| `State` | Indian state of residence |
| `Number_of_Referrals` | Number of other customers this person referred |
| `Tenure_in_Months` | How long they've been a customer |
| `Value_Deal` | Promotional deal enrolled in, if any (many customers have none) |
| `Phone_Service` | Has phone service (Yes/No) |
| `Multiple_Lines` | Has multiple phone lines |
| `Internet_Service` | Has internet service (Yes/No) |
| `Internet_Type` | DSL / Cable / Fiber Optic (blank if no internet) |
| `Online_Security` | Add-on service |
| `Online_Backup` | Add-on service |
| `Device_Protection_Plan` | Add-on service |
| `Premium_Support` | Add-on service |
| `Streaming_TV` | Add-on service |
| `Streaming_Movies` | Add-on service |
| `Streaming_Music` | Add-on service |
| `Unlimited_Data` | Add-on service |
| `Contract` | Month-to-Month / One Year / Two Year |
| `Paperless_Billing` | Yes/No |
| `Payment_Method` | Bank Withdrawal / Credit Card / Mailed Check |
| `Monthly_Charge` | Current monthly bill (can be negative — see cleaning notes: credits/promos) |
| `Total_Charges` | Cumulative charges since joining |
| `Total_Refunds` | Cumulative refunds issued |
| `Total_Extra_Data_Charges` | Extra data overage charges |
| `Total_Long_Distance_Charges` | Long-distance charges |
| `Total_Revenue` | Net revenue from the customer (Total_Charges − Total_Refunds + extras) |
| `Customer_Status` | Stayed / Churned / Joined → source of the `churn` target |
| `Churn_Category` | Reason bucket (leakage, reporting only) |
| `Churn_Reason` | Free-text reason (leakage, reporting only) |

## Engineered features (created in `src/feature_engineering.py`)

| Feature | Definition |
|---|---|
| `avg_monthly_revenue` | `Total_Revenue / Tenure_in_Months` (with tenure floored at 1) |
| `complaint_rate` proxy → `refund_rate` | `Total_Refunds / (Total_Charges + 1)` — proxy for dissatisfaction (no raw complaints table exists) |
| `services_subscribed` | Count of Yes-flags across the 8 add-on service columns |
| `has_internet` | 1 if `Internet_Service` == Yes |
| `is_month_to_month` | 1 if `Contract` == Month-to-Month (known strong churn driver) |
| `long_distance_share` | `Total_Long_Distance_Charges / (Total_Revenue + 1)` |
| `extra_data_share` | `Total_Extra_Data_Charges / (Total_Revenue + 1)` |
| `tenure_bucket` | Binned tenure: 0-6mo, 6-12mo, 1-2yr, 2-4yr, 4yr+ |
| `revenue_per_referral` | `Total_Revenue / (Number_of_Referrals + 1)` |

> Note: the source data has no complaint log, support-ticket table, or purchase-event history,
> so CLV/engagement-trend features described in the original brief (e.g. `spending_trend`,
> `support_interaction_rate`) are approximated using the closest available billing signals and
> labeled as proxies above, rather than invented from nothing.
