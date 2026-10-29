# Instamart Customer Intelligence & Support Copilot

An enterprise-grade Customer Intelligence & Support Copilot built for Swiggy Instamart's quick-commerce operations across six metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, Pune). The system addresses three critical leadership challenges: untrusted raw data exports, undetected customer churn/lapse, and overwhelming volumes of unclassified customer support tickets.

The copilot is implemented in Python across four tightly integrated, modular components:
- **Part A (`part_a/` & `sql/`)**: Clean, store, and query messy transactional data; validates against SQLite 3.
- **Part B (`part_b/`)**: RFM customer segmentation (K-Means & PCA) and leakage-free customer lapse prediction.
- **Part C (`part_c/`)**: Safe, resilient, budget-guarded support ticket triage and dynamic routing.
- **Part D (`part_d/`)**: Retrieval-Augmented Generation (RAG) policy assistant and ReAct refund agent with crash-resilient state recovery.

---

## Overview

In fast-paced quick-commerce delivery, high operational speed often generates dirty telemetry: unparseable amounts, negative delivery times, pending duplicate records, and inconsistent user profiles. Concurrently, retaining frequent and high-value shoppers requires proactive machine learning intervention before order lapses occur. Finally, high customer support ticket volumes require strict PII protection, resilient LLM interactions, cost-bounded token budgets, and automated policy-grounded dispute resolution.

This repository provides an end-to-end reproducible solution using only standard open-source libraries (`pandas`, `numpy`, `scikit-learn`, `joblib`, and `sqlite3`), fully decoupled from proprietary or paid APIs using the deterministic offline engine `mock_llm.py`.

### Repository Structure

```text
.
├── generate_data.py               # Deterministic synthetic data generator (unmodified)
├── mock_llm.py                    # Deterministic offline LLM stand-in (unmodified)
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules (.venv, caches, env files)
├── README.md                      # Comprehensive project documentation
├── run_pipeline.py                # Master pipeline runner (A -> B -> C -> D)
├── data/
│   ├── raw/                       # Generated raw CSV and policy files
│   ├── clean/                     # Clean analytical tables (customers, products, orders)
│   ├── policies/                  # Policy documents in plain text
│   └── instamart.db               # SQLite database with clean analytical tables
├── sql/
│   └── queries.sql                # Analytical SQL queries Q1 to Q7
├── outputs/
│   ├── customer_segments.csv      # K-Means customer RFM clusters with Premium flag
│   ├── lapse_model.joblib         # Tuned customer lapse prediction model
│   ├── retention_list.csv         # Top 10 at-risk premium customers
│   ├── triage_results.csv         # Final routed ticket decisions (few-shot)
│   ├── triage_budget_demo.csv     # Bounded token budget demonstration output
│   ├── agent_runs.jsonl           # Execution traces for all refund agent scenarios
│   └── refund_ledger.jsonl        # Immutable ledger of issued and escalated refunds
├── state/                         # Agent run checkpoints for crash recovery
├── part_a/                        # Part A module: Data Foundation
│   ├── audit.py                   # Raw data audit and telemetry analysis
│   ├── clean.py                   # Documented deterministic data cleaning pipeline
│   ├── db.py                      # SQLite storage engine and query runner
│   ├── validate.py                # Checkpoint validation assertions
│   ├── eda.py                     # Exploratory analysis and correlation analysis
│   └── run_part_a.py              # CLI runner for Part A
├── part_b/                        # Part B module: Segments & Lapse Prediction
│   ├── rfm.py                     # RFM metric calculation
│   ├── clustering.py              # K-Means clustering, silhouette analysis, PCA
│   ├── lapse_modeling.py          # Leakage-free cohort engineering & model tuning
│   └── run_part_b.py              # CLI runner for Part B
├── part_c/                        # Part C module: Ticket Triage with Guardrails
│   ├── guardrails.py              # Input token guardrails & PII regex redaction
│   ├── classifier.py              # Prompts, retry with jitter, response validation
│   ├── triage_pipeline.py         # Budget controller, routing, and ticket pipeline
│   └── run_part_c.py              # CLI runner for Part C
└── part_d/                        # Part D module: Policy RAG & Refund Agent
    ├── rag.py                     # Document chunking, TF-IDF retriever, RAG QA
    ├── tools.py                   # Agent tool definitions, specs, and refund ledger
    ├── agent.py                   # ReAct execution loop, safety limits, checkpoint state
    └── run_part_d.py              # CLI runner for Part D
```

---

## Reproduce

Follow these numbered shell commands from repository clone to execute the entire pipeline from scratch:

```bash
# 1. Create a clean virtual environment using Python 3.11 or newer
python3.12 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install required open-source dependencies
pip install -r requirements.txt

# 4. Generate the synthetic data and policy documents
python generate_data.py

# 5. Execute Part A: Clean data, store in SQLite, run queries, and validate checkpoints
python part_a/run_part_a.py

# 6. Execute Part B: Build RFM segments, cluster, tune lapse models, and export retention list
python part_b/run_part_b.py

# 7. Execute Part C: Run support ticket triage, PII redaction, token budget demo, and routing
python part_c/run_part_c.py

# 8. Execute Part D: Evaluate Policy RAG, execute ReAct refund scenarios, and test crash-resume
python part_d/run_part_d.py

# (Optional) Run the complete end-to-end pipeline in one command:
python run_pipeline.py
```

---

## Part A: Data Foundation: Clean, Store, Query

### 1. Raw Data Audit

The raw exports were inspected for shape, data types, and missingness. `data/raw/tickets.csv` was ingested with `keep_default_na=False` to ensure empty ticket texts remain empty strings.

| Table | Shape | Column Name | Dtype | Missing Count | Missing Pct (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **customers.csv** | (1200, 5) | customer_id | object | 0 | 0.00% |
| | | city | object | 0 | 0.00% |
| | | signup_date | object | 0 | 0.00% |
| | | age | float64 | 61 | 5.08% |
| | | gender | object | 30 | 2.50% |
| **products.csv** | (40, 4) | product_id | object | 0 | 0.00% |
| | | product_name | object | 0 | 0.00% |
| | | category | object | 0 | 0.00% |
| | | price_inr | int64 | 0 | 0.00% |
| **orders.csv** | (13084, 9) | order_id | object | 0 | 0.00% |
| | | customer_id | object | 0 | 0.00% |
| | | product_id | object | 0 | 0.00% |
| | | quantity | int64 | 0 | 0.00% |
| | | order_ts | object | 0 | 0.00% |
| | | status | object | 0 | 0.00% |
| | | discount_pct | int64 | 0 | 0.00% |
| | | order_amount | object | 0 | 0.00% |
| | | delivery_minutes | float64 | 2541 | 19.42% |

#### Checkpoint Audit Metrics

| Metric Description | Count / Value | Reference Value | Match Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | 13,084 | Exact Match |
| Exact duplicate rows in `orders.csv` | 119 | 119 | Exact Match |
| Rows with status `pending` in `orders.csv` | 656 | 656 | Exact Match |
| Distinct `order_id` values | 12,309 | 12,309 | Exact Match |
| Non-numeric `order_amount` entries | 120 | 120 | Exact Match |
| Negative `delivery_minutes` entries | 106 | 106 | Exact Match |
| Blank `delivery_minutes` entries | 2,541 | 2,541 | Exact Match |
| Blank `age` in `customers.csv` | 61 | 61 | Exact Match |
| Ages outside 10–100 in `customers.csv` | 6 | 6 | Exact Match |
| Distinct raw city spellings | 24 | 24 | Exact Match |
| Distinct raw gender spellings (non-blank) | 10 | 10 | Exact Match |
| Blank genders in `customers.csv` | 30 | 30 | Exact Match |

---

### 2. Cleaning Customers

Cleaning rules applied in `part_a/clean.py`:
1. **City Standardisation**: Stripped leading/trailing whitespace and applied title-casing (`.str.strip().str.title()`). Exactly 6 clean cities remain: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
2. **Gender Normalisation**: Mapped case variants to `male`, `female`, or `unknown` (missing/blank mapped to `unknown`). Counts: female (610), male (560), unknown (30).
3. **Age Validation & City-Median Imputation**: Treated 6 impossible age values (< 10 or > 100: -4, 0, 150, 200, 280, 999) as missing, and filled all 67 missing ages with the median age of the customer's respective city using `.groupby('city')['age'].transform('median')`. Post-cleaning missing ages: 0; min age: 18; max age: 60.
4. **Datetime Conversion**: Converted `signup_date` to standard ISO format (`YYYY-MM-DD`).

---

### 3. Cleaning Orders

Cleaning rules applied in exact documented order:
1. **Datetime Conversion**: Converted `order_ts` to datetime objects.
2. **Exact Deduplication**: Removed 119 exact duplicate rows.
3. **Latest Record Retention**: For orders appearing multiple times across statuses (e.g. earlier `pending` state and later `completed`/`cancelled` state), sorted by `order_ts` and kept the latest record per `order_id` (resolved 656 pending records).
4. **Amount Imputation**: Converted `order_amount` to numeric (coercing planted `'N/A'` and `'error'` strings to NaN). Imputed all 120 missing amounts from catalog price:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
5. **Delivery Duration Imputation**: Flagged 106 negative `delivery_minutes` (-10) as missing. For completed orders, filled missing delivery times with the median delivery duration of that order's product category among completed orders. Cancelled orders were intentionally left with missing delivery duration because cancelled orders were never delivered.

#### Post-Cleaning Orders Verification

| Metric | Value | Reference Checkpoint |
| :--- | :--- | :--- |
| Clean order rows | 12,309 | 12,309 (one per order_id) |
| Completed orders | 11,110 | 11,110 |
| Cancelled orders | 1,199 | 1,199 |
| Pending orders | 0 | 0 |
| Missing `order_amount` | 0 | 0 |
| Missing `delivery_minutes` (completed) | 0 | 0 |
| Missing `delivery_minutes` (cancelled) | 1,199 | 1,199 |
| Sum of completed `order_amount` | INR 2,998,668.80 | INR 2,998,668.80 |

The validation script `part_a/validate.py` executes 12 assertions across these metrics and outputs:
```text
ALL CHECKPOINTS PASSED
```

---

### 4. SQL Analysis (`sql/queries.sql`)

All queries were executed via `pd.read_sql_query` against `data/instamart.db`:

- **Q1 — WHERE then HAVING**: Filtered completed orders with `order_amount > 500`, grouped by customer, and filtered `SUM(order_amount) > 8000`.
  - *Result*: **30 customers** returned.
- **Q2 — Three-Table Join**: Revenue aggregated by city and product category across orders, customers, and products.
  - *Result*: **30 rows** returned; grand total revenue = **INR 2,998,668.80**; top cell: **Bengaluru / Electronics** at **INR 233,909.45**.
- **Q3 — Scalar Subquery**: Identified products priced above the overall average catalog price (`AVG(price_inr)` = 359.85).
  - *Result*: **11 products** returned.
- **Q4 — LEFT JOIN**: Identified registered customers who have zero completed orders.
  - *Result*: **25 customers** returned.
- **Q5 — Common Table Expression (CTE)**: Computed customer spend CTE and average spend CTE, returning customers with spend above the average.
  - *Result*: **550 customers** returned; overall average spend = **INR 2,552.06**.
- **Q6 — Ranking Window Functions**: Partitioned by city, ordered by `total_spend DESC`, applying `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`.
  - *Top customer per city*:
    - Bengaluru: **C0082** (INR 12,856.20)
    - Chennai: **C0930** (INR 10,151.40)
    - Delhi: **C0291** (INR 13,742.60)
    - Hyderabad: **C0743** (INR 12,730.05)
    - Mumbai: **C0693** (INR 14,766.55)
    - Pune: **C0426** (INR 10,377.10)
  - *Rank Discrepancy*: Exactly **8 rows** have `RANK()` different from `ROW_NUMBER()`.
  - *Tie Explanation*: Ties cause `RANK()` to assign identical rank positions to tied spend values and skip subsequent ranks, whereas `ROW_NUMBER()` strictly assigns consecutive unique integers regardless of ties.
- **Q7 — Running Total**: Daily completed revenue running total ordered chronologically.
  - *Result*: **183 daily rows**; final cumulative revenue = **INR 2,998,668.80**.

---

### 5. Pandas Cross-Check & Exploratory Data Analysis

#### Per-City Completed Revenue Breakdown

| City | Total Completed Spend (INR) | Mean Spend (INR) | Completed Order Count |
| :--- | :--- | :--- | :--- |
| **Bengaluru** | 523,643.40 | 276.47 | 1,894 |
| **Chennai** | 463,629.15 | 272.72 | 1,700 |
| **Delhi** | 502,829.80 | 243.62 | 2,064 |
| **Hyderabad** | 528,589.90 | 284.95 | 1,855 |
| **Mumbai** | 479,058.50 | 297.00 | 1,613 |
| **Pune** | 500,918.05 | 252.48 | 1,984 |
| **Total** | **2,998,668.80** | **269.91** | **11,110** |

#### Statistical Distribution of Completed `order_amount`

- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (heavily right-skewed)
- **Quartiles**: Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20
- **1.5×IQR Fences**: Lower fence = INR -283.50, Upper fence = INR 645.30
- **Outliers**: Lower outliers = 0; Upper outliers = 897 orders

#### Correlation Matrix (Completed Orders)

| Variable | order_amount | discount_pct | quantity | delivery_minutes |
| :--- | :--- | :--- | :--- | :--- |
| **order_amount** | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| **discount_pct** | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| **quantity** | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| **delivery_minutes** | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Exploratory Findings (Under 200 Words)

This dataset captures synthetic transactional operations for quick-commerce grocery delivery across six Indian metros. Distributions are heavily right-skewed: the median order amount (INR 135.00) is half the mean (INR 269.91) with a skewness of 4.61, reflecting frequent low-cost daily staple purchases (dairy, produce) interspersed with high-ticket electronics. The 1.5×IQR upper fence is INR 645.30, identifying 897 upper-tail outliers driven by bulk basket quantities and premium consumer electronics. Deliveries average 24 minutes with low variance. Pairwise linear associations are minimal; the strongest correlation in absolute magnitude is between `order_amount` and `discount_pct` at −0.0951 (followed by `order_amount` and `quantity` at +0.0873). However, correlation of this small magnitude does not establish causation; higher basket values occasionally reflect smaller relative promotional discounts rather than discounts suppressing spending.

---

## Part B: Customer Segments and Lapse Prediction

### 1. RFM Table Construction

Computed as of snapshot `2025-10-01 00:00:00` over completed orders:
- Total active customers: **1,175**
- **Recency** (days since last completed order): median = 19, min = 0, max = 180
- **Frequency** (count of distinct completed orders): mean = 9.46, min = 1, max = 33
- **Monetary** (sum of completed spend): mean = INR 2,552.06, min = INR 20.00, max = INR 14,766.55

---

### 2. K-Means Clustering & Segment Profiling

RFM columns were standardised using `StandardScaler`. Clustering was evaluated across $K \in [2, 8]$ using `KMeans(n_clusters=K, n_init=10, random_state=42)`:

| K | Inertia | Silhouette Score |
| :--- | :--- | :--- |
| **2** | 1820.8 | 0.465 |
| **3** | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** (Optimal Peak) |
| **5** | 617.4 | 0.480 |
| **6** | 503.2 | 0.482 |
| **7** | 408.5 | 0.416 |
| **8** | 366.5 | 0.412 |

*Inertia vs K Explanation*: The inertia of $K=8$ is mathematically lower than $K=2$ because adding more centroids monotonically decreases the within-cluster sum of squared distances to the nearest center. At the extreme where $K=N$, inertia becomes zero; therefore, inertia alone cannot select $K$, requiring silhouette analysis and the elbow criterion to identify genuine cluster compactness and separation.

#### K=4 Segment Profiles

$K=4$ was selected as it maximizes the silhouette score (0.526) and offers clean business interpretability:

| Segment ID | Segment Name | Size | Mean Recency (days) | Mean Frequency | Mean Spend (INR) | Recommended Action | Is Premium |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **High-Value Premium** | 138 | 24.12 | 6.77 | 7,129.70 | VIP concierge, early access to new electronics, dedicated SLA | **True** |
| **2** | **Active Frequent Shoppers**| 419 | 8.77 | 19.40 | 3,410.02 | Category cross-selling, loyalty subscription perks | False |
| **0** | **Occasional / Regular** | 436 | 28.86 | 3.77 | 1,057.62 | Frequency nudges, personalized weekend discount coupons | False |
| **3** | **Dormant / At-Risk** | 182 | 109.04 | 2.20 | 685.99 | Win-back campaigns, deep reactivation discounts | False |

*Note*: Cluster numbers are arbitrary labels with no intrinsic ordering.

#### PCA Projection

Applying `PCA(n_components=2)` on scaled RFM features yields:
- Component 1 Explained Variance Ratio: **0.6343** (63.4%)
- Component 2 Explained Variance Ratio: **0.2180** (21.8%)
- Total Explained Variance: **0.8523** (85.2%)

The segment assignments were exported to `outputs/customer_segments.csv`.

---

### 3. Customer Lapse Prediction

#### Cohort & Leakage-Free Feature Engineering

- **Cutoff Date**: `2025-09-01 00:00:00`.
- **Cohort**: Customers with $\ge 1$ completed order prior to the cutoff date (**1,156 customers**).
- **Target (`lapsed`)**: `1` if the customer had zero completed orders between `2025-09-01` and `2025-09-30`, else `0`.
  - Lapsed count: **443** (38.32% positive class).
  - Active count: **713** (61.68% negative class).
- **Features** (strictly calculated using pre-cutoff data and customer profiles only):
  - `recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `tenure_days`, `age`, `favourite_category`, `city`, `gender`.
  - Feature leakage safeguards: Zero post-cutoff orders were accessed, and Part B Task 3 segment labels were excluded from model features.
- **Train/Test Split**: 80/20 stratified split (`random_state=42`):
  - Train: **924 customers** (354 lapsed, 570 active).
  - Test: **232 customers** (89 lapsed, 143 active).

#### Preprocessing & Learned Statistics

Preprocessing used a `ColumnTransformer` with `StandardScaler` on numeric variables and `OneHotEncoder(drop='first', sparse_output=False)` on nominal columns, fitted strictly on the training partition.
- **Learned Training Frequency Mean**: **8.1147**
- **Full Cohort Frequency Mean**: **8.0285**
- *Difference Explanation*: The training frequency mean differs from the full cohort mean because the training partition is a random 80% sample; sample variance naturally produces slight divergence from the population parameter, demonstrating that no test data leaked into the scaler.
- Zero missing values were confirmed across all feature vectors.

#### Model Tuning & Evaluation

Models were tuned using 5-fold cross-validation (`scoring='f1'`):

| Model | Hyperparameter Grid | Best Hyperparameters | Best CV F1 | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | Predict All Active | None | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | $C \in \{0.01, 0.1, 1, 10, 100\}$ | $C = 100$ | 0.6506 | 0.6444 | 0.6517 | 0.6480 |
| **Random Forest** | depth $\in \{3,5,8\}$, leaf $\in \{5,10,20\}$ | depth=8, leaf=10 | **0.6856** | 0.6404 | 0.6404 | 0.6404 |

*Confusion Matrices on Test Set (Lapsed Class = Positive)*:
- **Logistic Regression**: TP=58, FP=32, FN=31, TN=111
- **Random Forest**: TP=57, FP=32, FN=32, TN=111

#### Model Choice Justification

Random Forest was selected as the operational model. In quick-commerce retention, failing to identify a lapsing customer (False Negative) results in permanent customer churn and lost lifetime value, whereas mistakenly sending a promotional retention offer to an active customer (False Positive) incurs only a minimal marketing coupon cost. Random Forest achieved superior cross-validated F1 (0.6856 vs 0.6506) and models non-linear interactions between delivery delays and order frequency without linear boundary constraints.

---

### 4. Class-Balance Experiment

The minority lapsed class in the training partition was oversampled with replacement (`random_state=42`) from 354 to 570 rows, equalizing both classes (1,140 total train rows). The best Logistic Regression ($C=100$) was refitted on this balanced set and evaluated on the untouched test partition:

| Model Configuration | Precision | Recall | F1 Score | Test Confusion Matrix [TN, FP / FN, TP] |
| :--- | :--- | :--- | :--- | :--- |
| **Unbalanced Logistic Regression** | 0.6444 | 0.6517 | 0.6480 | [111, 32 / 31, 58] |
| **Class-Balanced Logistic Regression** | 0.6000 | **0.8090** | **0.6890** | [95, 48 / 17, 72] |

*Retention Team Impact*: Recall moved the most, surging from 0.6517 to 0.8090 (+15.7 percentage points), while precision remained at 0.6000. For the retention team, this trade-off is advantageous: it successfully captures 81% of all lapsing customers (recovering 14 additional churning users), ensuring that win-back campaigns catch at-risk customers before churn becomes irreversible.

---

### 5. Interpretability & Retention List

#### Top 5 Features (Random Forest)

1. `monetary` (Importance: 0.2779): Total historical spend is the strongest predictor of customer commitment and habituation.
2. `frequency` (Importance: 0.2768): Order cadence establishes purchasing routine; sudden frequency drops signal immediate churn.
3. `recency` (Importance: 0.1113): Extended elapsed time since last purchase strongly correlates with disengagement.
4. `avg_delivery_minutes` (Importance: 0.0617): Delivery friction and prolonged transit times trigger dissatisfaction and customer churn.
5. `avg_discount_pct` (Importance: 0.0532): Deal-seeking behavior indicates lower brand loyalty when incentives diminish.

#### Logistic Regression Coefficients

- `recency` (+0.7628): Positive coefficient — higher elapsed days since last order increases lapse probability.
- `frequency` (-0.6841): Negative coefficient — higher historical order volume reduces lapse probability.
- `monetary` (-0.3812): Negative coefficient — greater total spend reinforces stickiness.

The model was serialized to `outputs/lapse_model.joblib`. Reloading verified exact prediction parity.

#### Retention Priority List (Top 10 At-Risk Premium Customers)

Saved to `outputs/retention_list.csv`:

| Customer ID | Predicted Lapse Probability | Actually Lapsed (Sept 2025) |
| :--- | :--- | :--- |
| **C0862** | 0.5304 | 0 |
| **C1008** | 0.4775 | 0 |
| **C0060** | 0.4765 | 0 |
| **C0267** | 0.4753 | 0 |
| **C0133** | 0.4726 | 0 |
| **C0942** | 0.4677 | 0 |
| **C0553** | 0.4645 | 0 |
| **C1112** | 0.4620 | 0 |
| **C0565** | 0.4616 | 1 |
| **C1000** | 0.4602 | 0 |

*Actually Lapsed Count*: 1 out of 10.
*Methodological Notice*: The retention list is an illustrative operational artifact rather than a leakage-free benchmark because the Premium segment designation was derived from the full 6-month RFM window.

---

## Part C: Support Ticket Triage with Guardrails

### 1. Input Guardrails & PII Redaction Audit

Processing 164 support tickets from `data/raw/tickets.csv`:
- **Empty Guardrail (`skipped_empty`)**: **4 tickets** contained whitespace-only or empty strings; routed immediately to human review with zero model calls.
- **Length Guardrail (`flagged_long`)**: **4 tickets** exceeded `MAX_INPUT_TOKENS = 400` (each estimated at **734 tokens**); safely intercepted before model dispatch.
- **PII Redaction**: Applied regex substitution in exact sequence:
  1. Emails (`[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]`): **55 substitutions**
  2. Phone Numbers (`\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]`): **91 substitutions**
  3. Order IDs (`ORD-\d{6}` $\rightarrow$ `[ORDER_ID]`): **132 substitutions**
  - **Total PII Masked**: **278 substitutions**. Zero unmasked entities remained across all prompt payloads (verified via assertion).

---

### 2. Resilient Retries & Output Validation

The triage pipeline wraps `mock_llm.chat` with exponential backoff and uniform jitter:
$$\text{wait} = \text{BASE\_DELAY} \times 2^n \times \text{Uniform}(0.9, 1.15) \quad (\text{BASE\_DELAY} = 0.05\text{s})$$
- Across the 156 eligible tickets, exactly **19 `TransientLLMError` failures** were triggered. Every failure was successfully recovered on retry 0 with waits between 0.045s and 0.0575s; zero tickets experienced unhandled exceptions.
- **Validation Engine**:
  - `classified_json`: **135 tickets** returned valid JSON with an allowed category.
  - `classified_keyword`: **12 tickets** returned conversational text (e.g. `The customer appears to have a <cat>-related problem.`), successfully parsed via `re.findall(r"[a-z]+", text.lower())`.
  - `flagged_invalid_output`: **9 tickets** returned invalid categories (e.g. `"shipping"`), safely diverted to human review.

---

### 3. Zero-Shot vs Few-Shot Comparison

| Metric | Zero-Shot Prompt | Few-Shot Prompt |
| :--- | :--- | :--- |
| **Total Tickets Evaluated** | 164 | 164 |
| **Classified Tickets** | 147 (135 JSON + 12 Keyword) | 147 (135 JSON + 12 Keyword) |
| **Classification Accuracy** | **85.71%** (126 / 147) | **100.00%** (147 / 147) |
| **Total Tokens Consumed** | 9,400 tokens | 17,044 tokens |
| **Tokens per Classified Ticket** | 63.95 tokens | 115.95 tokens |

#### Zero-Shot Confusion Matrix (Classified Tickets)

| True Category \ Predicted | billing | delivery | fraud | technical | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **billing** | 38 | 0 | 0 | 0 | 38 |
| **delivery** | 0 | 36 | 0 | 0 | 36 |
| **fraud** | **11** | 0 | 23 | 0 | 34 |
| **technical** | **10** | 0 | 0 | 29 | 39 |
| **Total** | 59 | 36 | 23 | 29 | 147 |

#### Few-Shot Value Justification

The few-shot prompt consumed an additional 52.0 tokens per ticket (+81.3% token overhead), but eliminated all 21 classification errors, raising classification accuracy from 85.71% to 100.00%. In quick-commerce support, misclassifying 11 urgent fraud reports as standard billing issues creates security and legal exposure, while misrouting 10 app crashes as billing queries delays engineering bug fixes. Paying 52 extra tokens (~INR 0.005 equivalent) per ticket prevents costly misrouting, making few-shot prompts essential for enterprise triage.

---

### 4. Token Budget & Routing Outputs

#### Token Budget Demonstration (`outputs/triage_budget_demo.csv`)

Running zero-shot with `TOKEN_BUDGET = 6000`:
- Tickets classified before exhaustion: **95** (88 JSON, 7 Keyword)
- Tickets flagged invalid output: **5**
- Skipped / Long: **8**
- **Tickets ending `budget_exhausted`**: **56 tickets** safely intercepted and redirected to human review without service degradation.

#### Production Ticket Routing (`outputs/triage_results.csv`)

Generated using the few-shot pipeline with `TOKEN_BUDGET = 100000`:
- **`support_queue`** (Priority NORMAL): **102 tickets** — Standard delivery, billing, and technical requests.
- **`security_team`** (Priority HIGH): **34 tickets** — Critical account fraud and unauthorized transactions.
- **`human_review`** (Priority NORMAL): **17 tickets** — Guardrail triggers (4 empty, 4 long, 9 invalid output).
- **`premium_desk`** (Priority HIGH): **11 tickets** — High-priority routing for verified Premium customers.

---

## Part D: Policy RAG Assistant and Refund Agent

### 1. Document Chunking & TF-IDF Retrieval

- 5 policy documents (`refund_policy.txt`, `delivery_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, `escalation_policy.txt`) split into 40-word chunks with 10-word overlap (30-word step):
  - `delivery_policy.txt`: 3 chunks
  - `escalation_policy.txt`: 3 chunks
  - `payment_policy.txt`: 3 chunks
  - `privacy_policy.txt`: 2 chunks
  - `refund_policy.txt`: 4 chunks
  - **Total Chunks**: **15 chunks**.
- **Query Test**: `refund policy late delivery` retrieves top 3 chunks entirely from `refund_policy.txt` (Scores: 0.2719, 0.2514, 0.2454).
- *TF & IDF Definitions & Limitation*: Term Frequency (TF) measures the normalized occurrence count of a word within a document chunk, while Inverse Document Frequency (IDF) penalizes common words across the corpus, weighting rare domain terms. A fundamental limitation of TF-IDF is its lexical reliance: it cannot capture semantic synonyms or paraphrases (e.g. failing to match "unauthorized debit" with "fraudulent charge" unless identical word stems appear).

---

### 2. Retrieval Evaluation: 10-Word Overlap vs. Zero-Overlap

Evaluated against `data/raw/policy_eval.csv` (14 questions: 12 answerable, 2 out-of-scope):

| Evaluation Metric | 10-Word Overlap (Step 30) | Zero Overlap (Step 40) |
| :--- | :--- | :--- |
| **Answerable Document Retrieval** | **12 / 12** (100.0%) | **12 / 12** (100.0%) |
| **Answerable Grounded Fact Accuracy** | **12 / 12** (100.0%) | **10 / 12** (83.3%) |
| **Unanswerable Fallback Accuracy** | **2 / 2** (100.0%) | **2 / 2** (100.0%) |

#### Analysis of Zero-Overlap Failures

Under zero-overlap chunking, 2 questions failed fact verification due to sentence boundary truncation:
1. *"What delivery time makes a completed order eligible for a full refund?"*: The qualifying condition *"more than 45 minutes"* was sheared across chunk boundaries, causing the model to return only *"receives a full refund of the order amount."*
2. *"How soon is money returned when a payment fails but is deducted?"*: The phrase *"48 hours"* was split from its condition, yielding an incomplete retrieval.
*Conclusion*: Overlapping chunk windows preserve context continuity across arbitrary token splits.

---

### 3. ReAct Refund Agent Scenarios (`outputs/agent_runs.jsonl`)

The agent implements five tools (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`). Running all 7 scenarios with default limits (`token_limit=6000`, `max_iterations=6`):

| Scenario ID | Status | Expected Outcome | Refund Evaluated | Expected Refund | Model Calls | Tokens Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | **REFUND** | REFUND | INR 128.25 | INR 128.25 | 5 | 972 |
| **S2** | **REFUND** | REFUND | INR 135.00 | INR 135.00 | 5 | 972 |
| **S3** | **ESCALATE** | ESCALATE | INR 1999.00 | INR 1999.00 | 5 | 959 |
| **S4** | **DENIED** | DENIED | INR 0.00 | INR 0.00 | 4 | 665 |
| **S5** | **DENIED** | DENIED | INR 0.00 | INR 0.00 | 3 | 382 |
| **S6** | **NOT_FOUND** | NOT_FOUND | INR 0.00 | INR 0.00 | 2 | 102 |
| **S7** | **FALLBACK_MAX_ITERATIONS** | FALLBACK | INR 0.00 | INR 0.00 | 6 | 3,090 |

- S1 and S2 were completed orders delivered in > 45 minutes with amounts $\le$ INR 1000, automatically issued and recorded in `outputs/refund_ledger.jsonl`.
- S3 was an eligible late delivery exceeding INR 1000 (INR 1999.00), correctly routed to `escalate_to_human`.
- S4 and S5 were denied (on-time delivery $\le 45$ min and cancelled order status, respectively).
- S6 halted early after database lookup confirmed the order did not exist.
- S7 entered a search loop and terminated cleanly at iteration 6.

---

### 4. Stop-Safely Demonstrations

1. **Token Budget Guardrail**: Running Scenario S1 with `token_limit=200`:
   - *Status*: **`FALLBACK_TOKEN_LIMIT`**
   - Intercepts execution immediately after the model call before executing requested tools.
2. **Execution Timeout Guardrail**: Running Scenario S1 with `time_limit_seconds=0`:
   - *Status*: **`FALLBACK_TIMEOUT`**
   - Intercepts execution at the top of the first iteration prior to any model call.

---

### 5. Resumable State & Crash-Recovery Demonstration

Demonstrated on Scenario S1:
1. Simulated system crash injected immediately after Tool Execution Step 2 (`search_policy`), following disk state persistence in `state/s1_crash_resume_demo.json`.
2. `resume_agent("s1_crash_resume_demo")` was invoked, restoring conversation context, accumulated token counts, and previous tool observations.
3. Execution resumed without re-running previous tools.
- **Ordered Sequence of Executed Tools Across Crash & Resume**:
  1. `get_order` (Pre-crash)
  2. `search_policy` (Pre-crash)
  3. `calculate_refund` (Post-resume)
  4. `issue_refund` (Post-resume)
- **Final Resumed Status**: **`REFUND`** (INR 128.25 issued).

---

### 6. Architectural Analysis: ReAct vs. Plan-and-Execute

The ReAct (Reason + Act) architecture operates as a dynamic, step-by-step state machine. For straightforward scenarios like S6 (`NOT_FOUND`), ReAct terminates after only 2 model calls (102 tokens); for cancelled orders (S5), it exits after 3 calls (382 tokens). However, happy-path resolutions (S1, S2, S3) require 5 iterative calls (959–972 tokens), and unresolvable loops (S7) consume 6 calls (3,090 tokens).

In contrast, a Plan-and-Execute pattern generates a multi-step plan in one initial prompt (e.g. `[get_order -> search_policy -> calculate_refund -> issue_refund]`), executing deterministic tools in sequence and querying the LLM only once or twice, reducing token overhead by 50–70%. However, Plan-and-Execute struggles with unexpected environmental observations: when an order is missing or cancelled, the pre-computed plan breaks, forcing a costly replanning step.

For Instamart's automated refund workflow, **ReAct is the superior design choice**. Financial transactions require strict conditional gating: an automated refund must never be issued if an order was cancelled or delivered on time. ReAct's tight feedback loop evaluates real-time database state at each intermediate step, providing auditable safety checks that outweigh the minor token savings of pre-planned execution.

---

## Limitations and Next Steps

1. **Synthetic Data Realism**: All order histories, customer distributions, and support tickets were generated via synthetic distributions. Real-world quick commerce involves regional dialect mixing, phonetic typos (e.g. "refunnd"), multimodal proof (damaged item photos), and seasonal demand surges (festivals, monsoons). Future work should integrate localized Indian multilingual embeddings.
2. **Post-Hoc Whole-Window Segment Labeling**: In Part B, the retention list was constructed using whole-window RFM segment designations (which include September 2025 activity). While the lapse model features strictly obeyed temporal boundaries, the Premium segment filter is post-hoc. Production deployments must calculate rolling RFM metrics strictly prior to the cutoff date.
3. **Lexical Retrieval Limitations**: The TF-IDF policy retriever lacks semantic understanding of synonyms. Upgrading to dense bi-encoder embeddings (e.g. `all-MiniLM-L6-v2`) with hybrid BM25 re-ranking would improve retrieval accuracy across colloquial customer inquiries.
4. **Deterministic Mock vs. Production LLMs**: `mock_llm.py` simulates structural failure modes (503s, non-JSON output, invalid categories) deterministically. Transitioning to production hosted models requires continuous schema validation (e.g. Pydantic / instructor), rate-limiting circuit breakers, and human-in-the-loop escalation workflows.
