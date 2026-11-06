# Instamart Customer Intelligence & Support Copilot

## Overview

Complete offline solution for the four connected capstone parts. All data, policies and prices are synthetic and fictional; the brand name only frames the assignment. All monetary amounts use INR. Use Python 3.12 and SQLite 3.25 or newer. The pinned scientific packages require Python 3.12 or newer; Python 3.12 is the tested environment.

Part A cleans the exports and creates the analytical database. Part B consumes the clean CSVs to segment customers and predict lapse. Part C uses Part B segments to route privacy-masked support tickets. Part D queries Part A's database and reuses policy retrieval for a bounded, resumable refund workflow. All language-model calls go through the unmodified `mock_llm.py`; no API key, account or network connection is needed to run the project once its Python dependencies are installed.

The four named folders each contain a complete notebook with explanations, stepwise executable code, actual text outputs and assertions:

- [Part A notebook](part_a/Part_A_Data_Foundation.ipynb): audit, cleaning, SQLite, seven SQL queries and exploratory statistics.
- [Part B notebook](part_b/Part_B_Customer_Segments_and_Lapse_Prediction.ipynb): RFM, clustering, PCA, feature engineering, tuning, balancing, interpretation and retention.
- [Part C notebook](part_c/Part_C_Support_Ticket_Triage_with_Guardrails.ipynb): input checks, privacy, prompts, retries, validation, budget and routing.
- [Part D notebook](part_d/Part_D_Policy_RAG_and_Refund_Agent.ipynb): policy retrieval, evaluation, tools, scenarios, safety limits and crash recovery.

```text
.
├── README.md                         # All findings and reproduction instructions
├── generate_data.py                  # Supplied, unmodified deterministic generator
├── mock_llm.py                       # Supplied, unmodified offline model
├── requirements.txt                  # Combined analytics and Jupyter environment
├── .gitignore
├── run_pipeline.py                   # All four parts, optionally regenerating raw data
├── part_a/                           # Data Foundation: sources and Part A notebook
├── part_b/                           # Segments and Lapse: sources and Part B notebook
├── part_c/                           # Ticket Triage: sources and Part C notebook
├── part_d/                           # Policy RAG and Refunds: sources and Part D notebook
├── sql/queries.sql                   # Q1-Q7 SQL
├── data/
│   ├── raw/                          # Six generated CSV inputs
│   ├── clean/                        # customers.csv, products.csv, orders.csv
│   ├── policies/                     # Five policy .txt documents
│   └── instamart.db                   # customers, products, orders tables
├── outputs/
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── agent_runs.jsonl              # Exactly seven current scenario runs
│   ├── agent_demos.jsonl             # Token, timeout and crash/resume results
│   ├── refund_ledger.jsonl           # Local simulated actions, with timestamps
│   ├── part_a_report.json ... part_d_report.json
│   ├── sql_q1.csv ... sql_q7.csv
│   ├── environment_report.json       # Installed pins and missing notebook tools
│   ├── notebook_execution_report.json
│   ├── reproducibility_report.json   # Separate clean-directory reproduction evidence
│   └── verification_report.json
├── state/                            # Saved agent checkpoints
└── scripts/
    ├── check_environment.py          # Python, SQLite and combined dependency checks
    ├── run_notebooks.py              # Cell execution with output and source verification
    ├── write_readme.py               # Rebuild this README from measured results
    └── verify_project.py             # Reference, artifact and regression checks
```

The original setup already uses branch `main`. Submission must use the public repository's `main` branch and include the regenerated artifacts and notebooks. The submission URL is `https://github.com/htg-pro/instamart-copilot-capstone/tree/main`; local changes must be committed and pushed before that URL includes them. No charts, screenshots, PDFs, slides, video or audio are generated or required.

## Reproduce

From the repository root, with Python 3.12 available as `python`, run these numbered commands in order. Dependency installation needs access to a package index or a local wheel cache; execution itself is offline.

```bash
# 1. Create an isolated environment
python -m venv .venv

# 2. Activate it (Linux/macOS)
source .venv/bin/activate

# 3. Install analytics and notebook tools together
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py

# 4. Generate the unmodified supplied data
python generate_data.py

# 5. Audit, clean, store, query and validate Part A
python part_a/run_part_a.py

# 6. Segment, tune, evaluate and persist Part B
python part_b/run_part_b.py

# 7. Compare prompts, demonstrate the budget and export Part C
python part_c/run_part_c.py

# 8. Evaluate policies, run seven cases and demonstrate recovery in Part D
python part_d/run_part_d.py

# 9. Execute all notebook cells and save actual text outputs
python scripts/run_notebooks.py

# 10. Refresh the README from those measured results
python scripts/write_readme.py

# 11. Verify required artifacts, references, regressions and notebook outputs
python scripts/verify_project.py
```

The generator's last line is exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Part A's standalone validation prints `ALL CHECKPOINTS PASSED`; the final project verifier prints `ALL PROJECT CHECKS PASSED`. `requirements.txt` combines the scientific packages used for these results and the pinned Jupyter tools; there is one dependency installation for the entire project. `.gitignore` excludes `.env`, `.venv/`, `__pycache__/` and notebook editor checkpoints.

Alternatively, `python run_pipeline.py --generate` performs steps 4-8. Use `python run_pipeline.py` if the supplied raw files already exist. Notebook execution reruns the respective part workflows and refreshes the same output paths. Batch agent logs and named checkpoints are reproducible; timestamps and elapsed times naturally vary, and unrelated old ledger/checkpoint records are preserved.

The notebook execution report records the Python runtime, execution method and SHA-256 fingerprints of notebook sources, saved outputs and project source files. The final verifier rejects stale evidence after changes to code, SQL, dependency pins or notebooks; rerun steps 9-11 after those edits. Official notebook schema validation also runs when the installed environment includes `nbformat`.

For interactive notebooks, run `python -m notebook`; Jupyter and its Python kernel are included in the same `requirements.txt`. In VS Code, select `.venv/bin/python` as the notebook kernel. Run all cells from top to bottom in A, B, C, D order. The bundled notebook executor supports these plain-Python, text-output notebooks; it is not a general executor for IPython magics or widget notebooks. On Windows, use `.venv\Scripts\activate` in Command Prompt for step 2.

Verification scope: the saved results were produced in the pinned analytics environment using the plain-Python notebook executor. Interactive Jupyter installation and launch have not been verified in this environment. The combined requirements file lists those tools, and `scripts/check_environment.py` checks their installation before a fresh reproduction.

## Part A

### Raw audit

Pandas reads the three raw analytical tables. Tickets are read with `keep_default_na=False` wherever they are used, preserving empty text. Missing percentages and actual data types are reported below; pandas parses `N/A` as missing, while `error` remains a nonnumeric string. Both count in the separate 120-value nonnumeric-amount audit.

| Table | Shape | Column | dtype | Missing | Missing % |
| --- | --- | --- | --- | --- | --- |
| customers | 1200 × 5 | customer_id | str | 0 | 0.00 |
| customers | 1200 × 5 | city | str | 0 | 0.00 |
| customers | 1200 × 5 | signup_date | str | 0 | 0.00 |
| customers | 1200 × 5 | age | float64 | 61 | 5.08 |
| customers | 1200 × 5 | gender | str | 30 | 2.50 |
| products | 40 × 4 | product_id | str | 0 | 0.00 |
| products | 40 × 4 | product_name | str | 0 | 0.00 |
| products | 40 × 4 | category | str | 0 | 0.00 |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00 |
| orders | 13084 × 9 | order_id | str | 0 | 0.00 |
| orders | 13084 × 9 | customer_id | str | 0 | 0.00 |
| orders | 13084 × 9 | product_id | str | 0 | 0.00 |
| orders | 13084 × 9 | quantity | int64 | 0 | 0.00 |
| orders | 13084 × 9 | order_ts | str | 0 | 0.00 |
| orders | 13084 × 9 | status | str | 0 | 0.00 |
| orders | 13084 × 9 | discount_pct | int64 | 0 | 0.00 |
| orders | 13084 × 9 | order_amount | str | 53 | 0.41 |
| orders | 13084 × 9 | delivery_minutes | float64 | 2541 | 19.42 |

| Raw audit checkpoint | Measured count |
| --- | --- |
| raw_order_rows | 13084 |
| exact_duplicates | 119 |
| pending_rows | 656 |
| distinct_order_ids | 12309 |
| non_numeric_amounts | 120 |
| negative_delivery_times | 106 |
| blank_delivery_times | 2541 |
| blank_ages | 61 |
| ages_outside_range | 6 |
| distinct_raw_cities | 24 |
| distinct_raw_genders | 10 |
| blank_genders | 30 |

### Cleaning rules and actual changes

Cities are stripped and title-cased; genders map to male/female/unknown. Ages outside 10-100 become missing, then all missing ages use the valid city median from `groupby` and `transform`. Fractional medians are preserved. Signup dates become datetimes.

Orders follow the required sequence: parse timestamps; remove exact duplicates; select the entire latest row per order id; parse and reconstruct missing amounts from `round(price_inr * quantity * (1 - discount_pct / 100), 2)`; replace negative delivery durations with missing; impute missing completed durations by completed-product-category median; leave cancelled durations missing. Selecting a whole row avoids accidentally borrowing a nonmissing field from an older record. Products are unchanged.

| Rule / stage | Rows or values affected |
| --- | --- |
| city_values_standardized | 476 |
| gender_values_standardized | 950 |
| blank_gender_to_unknown | 30 |
| invalid_ages_to_missing | 6 |
| ages_imputed_city_median | 67 |
| signup_dates_parsed | 1200 |
| order_timestamps_parsed | 13084 |
| exact_duplicate_rows_removed | 119 |
| older_order_records_removed | 656 |
| amounts_reconstructed | 112 |
| negative_delivery_to_missing | 105 |
| completed_delivery_imputed | 774 |
| cancelled_delivery_left_missing | 1199 |
| products_changed | 0 |

Raw amounts/negative durations include duplicates; therefore 120 raw nonnumeric amounts become 112 reconstructions, and 106 raw negative durations become 105 replacements after deduplication. The 774 completed-duration imputations include blanks and invalid durations.

| City | Age imputation median |
| --- | --- |
| Bengaluru | 42.0 |
| Chennai | 39.0 |
| Delhi | 43.0 |
| Hyderabad | 37.5 |
| Mumbai | 38.5 |
| Pune | 39.0 |

| Category | Completed delivery imputation median (minutes) |
| --- | --- |
| Dairy | 25.0 |
| Electronics | 25.0 |
| Grocery | 25.0 |
| Personal Care | 25.5 |
| Snacks | 25.0 |

| City | Clean customer count |
| --- | --- |
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Clean customer count |
| --- | --- |
| female | 610 |
| male | 560 |
| unknown | 30 |

### Storage and validation

Clean tables are written to `data/clean/` and SQLite tables `customers`, `products`, `orders` in `data/instamart.db`. Timestamps are stored as `YYYY-MM-DD HH:MM:SS`. `part_a/validate.py` checks customer distributions, valid ages, order identity/status counts, missingness and revenue; `part_a/report.py` additionally checks every SQL reference.

| Clean checkpoint | Measured result |
| --- | --- |
| Customers / products / orders | 1,200 / 40 / 12,309 |
| Completed / cancelled / pending | 11,110 / 1,199 / 0 |
| Unique order ids | 12,309 |
| Missing ages / age range | 0 / 18-60 |
| Missing amounts | 0 |
| Missing completed / cancelled delivery | 0 / 1,199 |
| Completed revenue | INR 2,998,668.80 |
| Standalone validator | ALL CHECKPOINTS PASSED |

### SQL business questions

All seven statements in `sql/queries.sql` run with `pd.read_sql_query`. Revenue and spend include completed orders only. Results are also exported as `outputs/sql_q1.csv` through `sql_q7.csv`.

| Query | Technique and result |
| --- | --- |
| Q1 | WHERE amount > INR 500, HAVING filtered spend > INR 8,000: 30 customers |
| Q2 | Three-table city/category join: 30 rows, total INR 2,998,668.80; largest Bengaluru/Electronics INR 233,909.45 |
| Q3 | Scalar subquery, price above catalogue average: 11 products |
| Q4 | LEFT JOIN, no completed orders: 25 customers |
| Q5 | Two CTEs, spend above customer average: 550 customers; average INR 2,552.06 |
| Q6 | City RANK/DENSE_RANK/ROW_NUMBER: 1,175 rows; 8 rank/row-number differences |
| Q7 | Daily revenue with cumulative SUM OVER: 183 days; final INR 2,998,668.80 |

| City | Top customer | Completed spend (INR) |
| --- | --- | --- |
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

Ties share RANK values while ROW_NUMBER assigns distinct consecutive positions, so eight rows have different rank and row-number values.

### pandas cross-check and exploratory statistics

The city/category pivot uses `pd.pivot_table(..., margins=True)` and its INR 2,998,668.80 grand total equals Q2. Named aggregation computes city sum, mean and count:

| City | Sum (INR) | Mean (INR) | Completed count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.80 | 243.62 | 2064 |
| Hyderabad | 528,589.90 | 284.95 | 1855 |
| Mumbai | 479,058.50 | 297.00 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

| Amount statistic | Measured result |
| --- | --- |
| mean | 269.91 |
| median | 135.00 |
| skewness | 4.61 |
| q1 | 64.80 |
| q3 | 297.00 |
| iqr | 232.20 |
| lower_fence | -283.50 |
| upper_fence | 645.30 |
| lower_outliers | 0 |
| upper_outliers | 897 |

Correlation matrix over completed orders:

| Feature | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| discount_pct | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| quantity | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| delivery_minutes | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

Exploratory interpretation (under 200 words): The data describes synthetic quick-commerce orders for 1,200 customers across six cities during April-September 2025. Completed revenue is INR 2,998,668.80 across 11,110 orders. Mean order amount is INR 269.91 versus median INR 135.00, with skewness 4.61: expensive baskets stretch the right tail. Q1 is INR 64.80 and Q3 is INR 297.00; IQR fences are INR -283.50 and INR 645.30. There are 897 upper outliers and no lower outliers. High-value electronics baskets need contextual review rather than automatic removal, while planted invalid ages, malformed amounts and negative delivery times need explicit cleaning. Amount and discount have the largest absolute correlation, about -0.095, followed by amount and quantity at +0.087. Both are weak associations; correlation of this size does not establish causation. City/category revenue and customer activity support the later segmentation and retention workflow.

## Part B

### RFM, clustering and business segments

Only completed orders enter RFM, as of `2025-10-01 00:00:00`. Recency uses whole elapsed days, frequency counts distinct orders, and monetary sums completed INR spend.

| RFM statistic | Measured value |
| --- | --- |
| customers | 1175 |
| recency_median | 19.00 |
| recency_min | 0 |
| recency_max | 180 |
| frequency_mean | 9.46 |
| frequency_max | 33 |
| monetary_mean_inr | 2,552.06 |
| monetary_max_inr | 14,766.55 |

The RFM columns are standardized with StandardScaler. Every K from 2-8 uses `KMeans(n_clusters=K, n_init=10, random_state=42)`:

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Choose K=4: the large inertia reduction from K=3 to K=4 is followed by smaller improvements, and K=4 has the strongest silhouette among the actionable choices K=3,4,5. Its four profiles support distinct occasional-buyer, premium, frequent-shopper and win-back campaigns. K-means inertia is the within-cluster squared-distance objective; adding centres increases flexibility, so the best achievable inertia cannot increase and the observed K=8 value is lower than K=2. Lower inertia alone cannot choose K because ever-smaller clusters may be less interpretable and less well separated. Cluster numbers are arbitrary labels and have no numerical ordering.

| ID | Business segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action | Premium |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Occasional / Regular Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Suggest relevant replenishment bundles to increase purchase frequency. | False |
| 1 | High-Value Premium | 138 | 24.12 | 6.77 | 7,129.70 | Offer a priority support desk and personalised high-value bundles. | True |
| 2 | Active Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | Reward repeat purchases with loyalty benefits and replenishment reminders. | False |
| 3 | Dormant / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Send a time-limited win-back offer and investigate service problems. | False |

PCA with two components explains 0.634279 and 0.218038, total 0.852316. `outputs/customer_segments.csv` has customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium; Premium is true only for the highest-mean-monetary segment.

### Lapse cohort, features and training isolation

Cutoff: `2025-09-01 00:00:00`. The cohort contains customers with at least one completed order before the cutoff. `lapsed=1` means no completed order during September 1-30; the target window ends exclusively at October 1. Orders after September cannot change this label.

Features are recency, frequency, monetary, mean completed discount, final-order cancellation rate, mean completed delivery minutes, favourite product category, city, age, gender and tenure days. Cancellation rate divides cancelled orders by completed plus cancelled orders before cutoff. The latest order entering features is `2025-08-31 22:35:45`. No order from September 1 onwards enters a feature, and no whole-window segment label is a feature. Code asserts no missing feature values.

`train_test_split(test_size=0.2, stratify=y, random_state=42)` is shared by both models. Numeric scaling and `OneHotEncoder(drop='first', handle_unknown='ignore')` live inside each Pipeline; GridSearchCV fits them separately on training folds. No encoder, scaler or resampler sees test rows. Favourite category, city and gender each use k-1 nominal columns.

| Cohort / split checkpoint | Value |
| --- | --- |
| Cohort | 1156 |
| Lapsed | 443 (38.32%) |
| Train / test | 924 / 232 |
| Training not-lapsed / lapsed | 570 / 354 |
| Test not-lapsed / lapsed | 143 / 89 |
| Fitted training frequency mean | 8.114719 |
| Full-cohort frequency mean | 8.028547 |
| Missing features | 0 |

The scaler mean uses training customers only; the full-cohort mean also includes held-out customers, so random sampling makes them differ.

### Tune on training data, then evaluate the untouched test set

LogisticRegression uses max_iter=1000, C in {0.01, 0.1, 1, 10, 100}, five-fold CV and F1 scoring: 25 CV fits plus refit. RandomForestClassifier uses 100 trees, random_state=42, max_depth in {3, 5, 8}, min_samples_leaf in {5, 10, 20}, five-fold CV and F1 scoring: 45 CV fits plus refit. Both searches finish before test evaluation. Confusion matrices use rows true [0,1], columns predicted [0,1], in the form [[TN,FP],[FN,TP]]. All precision, recall and F1 values concern lapsed=1; zero_division=0 is used.

| Model | Best parameters | Best CV F1 |
| --- | --- | --- |
| Logistic regression | {"clf__C": 100} | 0.6506 |
| Random forest | {"clf__max_depth": 8, "clf__min_samples_leaf": 10} | 0.6865 |

| Test model | Confusion matrix | Accuracy | Precision (1) | Recall (1) | F1 (1) |
| --- | --- | --- | --- | --- | --- |
| Always not lapsed | [[143, 0], [89, 0]] | 0.6164 | 0.0000 | 0.0000 | 0.0000 |
| Tuned logistic regression | [[111, 32], [31, 58]] | 0.7284 | 0.6444 | 0.6517 | 0.6480 |
| Tuned random forest | [[111, 32], [32, 57]] | 0.7241 | 0.6404 | 0.6404 | 0.6404 |

We select the tuned random forest because its training cross-validated F1 is highest (0.6865); the test set did not choose its parameters. On the untouched test set its lapsed-class F1 is 0.6404, precision is 0.6404 and recall is 0.6404. It misses 32 lapsing customers and wrongly targets 32 loyal customers at the default threshold. Missing a lapsing customer can lose repeat revenue, whereas an unnecessary retention offer costs discount spend and may annoy a loyal customer, so the team should validate campaign economics and a recall-oriented threshold on a separate validation set before deployment.
Measured best C can differ from the reference's example C=1; the specified grid, cohort and positive-class performance criteria are preserved.

### Class balance experiment

Keep every original training row, then sample lapsed training rows with replacement and random_state=42 until both classes have 570 rows. This adds 216 duplicates to the 924 original rows. A cloned best logistic pipeline is refitted on the balanced training data; the original test set and tuned base models remain untouched.

| Logistic training | Test precision (1) | Test recall (1) | Test F1 (1) |
| --- | --- | --- | --- |
| Original 570/354 | 0.6444 | 0.6517 | 0.6480 |
| Balanced 570/570 | 0.5891 | 0.8539 | 0.6972 |

Recall moves most (+0.2022); balancing increases the incentive to identify lapsing customers, but extra targeting can lower precision.

### Interpretation, persistence and retention export

Five largest random-forest feature importances:

| Feature | Importance | Business reading |
| --- | --- | --- |
| monetary | 0.277830 | Earlier spending captures customer value and purchase intensity in INR. |
| frequency | 0.276787 | Earlier completed-order frequency distinguishes regular shoppers. |
| recency | 0.111219 | Days since the last completed order indicate disengagement. |
| avg_delivery_minutes | 0.061649 | Earlier delivery duration captures service experience. |
| avg_discount_pct | 0.053178 | Earlier discount usage helps identify promotion-sensitive shoppers. |

Three fitted logistic coefficients (numeric coefficients refer to standardized inputs):

| Feature | Coefficient | Sign | Business reading |
| --- | --- | --- | --- |
| recency | 0.039585 | positive | Days since the last completed order indicate disengagement. Its coefficient implies higher lapse log-odds as the feature increases, holding other inputs fixed. |
| frequency | -1.698239 | negative | Earlier completed-order frequency distinguishes regular shoppers. Its coefficient implies lower lapse log-odds as the feature increases, holding other inputs fixed. |
| avg_discount_pct | 0.004438 | positive | Earlier discount usage helps identify promotion-sensitive shoppers. Its coefficient implies higher lapse log-odds as the feature increases, holding other inputs fixed. |

The selected random forest Pipeline is persisted to `outputs/lapse_model.joblib`; joblib reload reproduces original test predictions exactly: **True**. There are 40 Premium test customers; the top ten by probability are exported with the required columns:

| Customer | Lapse probability | Actually lapsed |
| --- | --- | --- |
| C0862 | 0.530369 | 0 |
| C0060 | 0.476473 | 0 |
| C1008 | 0.475621 | 0 |
| C0267 | 0.475263 | 0 |
| C0133 | 0.471179 | 0 |
| C0942 | 0.468798 | 0 |
| C1000 | 0.465437 | 0 |
| C0553 | 0.464462 | 0 |
| C1112 | 0.462014 | 0 |
| C0565 | 0.460255 | 1 |

Actually lapsed among the top ten: **1/10**. The Premium flag is a post-hoc whole-window RFM label that includes September; therefore the retention list is illustrative and is not a leakage-free evaluation.

## Part C

### Guardrails and privacy

Process tickets in raw CSV order. Empty or whitespace-only text is skipped; text with est_tokens > 400 is flagged long. Neither triggers a model call. `est_tokens` is the unmodified mock's ceil(words * 1.3) estimator.

| Input check | Measured result |
| --- | --- |
| Total tickets | 164 |
| Eligible | 156 |
| skipped_empty | 4 |
| flagged_long | 4 |
| Tokens per long ticket | 734 |

Apply exactly these masks, in this order, before building a prompt:

```text
email:    [\w.+-]+@[\w-]+\.[\w.]+  -> [EMAIL]
phone:    \b[6-9]\d{9}\b            -> [PHONE]
order id: ORD-\d{6}                  -> [ORDER_ID]
```

| PII type | Substitutions across all tickets | Remaining |
| --- | --- | --- |
| email | 55 | 0 |
| phone | 91 | 0 |
| order_id | 132 | 0 |
| Total | 278 | 0 |

Reapplying all three regexes finds zero matches. An assertion checks every prompt, including retry attempts. The user message is exactly the redacted ticket text, without appended labels or identifiers. The notebook additionally spies on all 175 attempted calls in a complete batch to prove privacy and file order.

### Prompts, retries and output validation

Zero-shot system prompt:

```text
CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
You must respond strictly in valid JSON format:
{"category": "...", "reason": "..."}
```

Few-shot system prompt:

```text
CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
You must respond strictly in valid JSON format:
{"category": "...", "reason": "..."}
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment.
Response: {"category": "billing", "reason": "charged twice for order"}
Ticket: Someone hacked my account and made unauthorized transactions.
Response: {"category": "fraud", "reason": "unauthorized account takeover"}
```

Each independent run immediately resets mock counters. Every call has at most three retries; wait for retry n=0,1,2 is `0.05 * 2**n * uniform(0.9,1.15)`, using a per-run random generator seeded 42. Complete runs encounter 19 transient failures, all recovered, with zero permanent errors. Every mock retry has n=0, so waits fall in 0.045-0.0575 seconds. One actual zero-shot retry log line appears below; all 19 are retained in `outputs/part_c_report.json` and notebook outputs.

```text
[RETRY] Ticket: T0002 | Retry: 0 | Wait: 0.052993s
```

Accept JSON only when its category is one of billing/delivery/technical/fraud (`classified_json`). Only a non-JSON answer may recover by `re.findall(r'[a-z]+', text.lower())`, and exactly one category word must appear (`classified_keyword`). Valid JSON with an invalid category, ambiguous text and every other invalid output become `flagged_invalid_output` and go to human review. No category is guessed from an invalid JSON reason.

| Full-run status | Count |
| --- | --- |
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Prompt comparison and token cost

Accuracy uses only classified tickets; tokens sum usage.total_tokens over all successful calls, including replies that fail validation. There is no token budget for these two comparison runs.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens / classified |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | 147 | 126 | 85.71% | 9400 | 63.946 |
| Few-shot | 147 | 147 | 100.00% | 17044 | 115.946 |

Zero-shot confusion table: rows are true category, columns predicted category; excluded invalid/guarded tickets are not in the denominator.

| True / predicted | billing | delivery | technical | fraud |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt increases tokens from 9,400 to 17,044, adding 7,644 tokens and about 52 per classified ticket. It corrects all 21 zero-shot errors: 11 fraud tickets and 10 technical tickets previously classified as billing. The extra tokens are worthwhile when a fraud misroute or delayed technical response costs more than the added processing budget. This synthetic mock comparison is perfect at 147/147, but live language-model quality would need separate evaluation.

### Budget and segment-aware routing

Before every model attempt, including a retry, stop if tokens already used are at least the budget. This assignment-prescribed gate is soft: the last admitted response may exceed the threshold. The zero-shot budget demo uses 6,000, exhausts 56 tickets, and ends at 6,038 tokens (38 over the gate). The final few-shot run uses exactly 100,000 and consumes 17,044 with no budget-exhausted rows.

Skipped, long, invalid or budget-exhausted tickets go to human_review/NORMAL. Other fraud tickets go to security_team/HIGH. Other Premium-customer tickets go to premium_desk/HIGH. All other classified tickets go to support_queue/NORMAL. Missing segment rows mean not Premium; category and segment_name remain blank when unavailable. Final CSV columns are ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used.

| Final route | Tickets |
| --- | --- |
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

| Final priority | Tickets |
| --- | --- |
| NORMAL | 119 |
| HIGH | 45 |

All 147 accepted outputs were rule-validated (135 JSON and 12 keyword); 17 guarded/invalid tickets went to humans. `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv` each contain all 164 input tickets.

## Part D

### Policy chunking, retrieval and grounded answers

Split each policy into 40-word chunks with 10-word overlap, never crossing document boundaries. Retain each chunk's source. Fit the prescribed default `TfidfVectorizer()` on the 15 chunks and retrieve the three highest cosine similarities, returning source, score and text. All three top chunks for `refund policy late delivery` come from refund_policy.txt.

TF measures how frequently a term appears within a chunk. IDF downweights terms common across many chunks and emphasizes rare terms, but lexical TF-IDF vectors can miss equivalent policy questions phrased with different words.

| Policy document | Chunks |
| --- | --- |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |

| Reference retrieval rank | Source | Cosine score |
| --- | --- | --- |
| 1 | refund_policy.txt | 0.271928 |
| 2 | refund_policy.txt | 0.251374 |
| 3 | refund_policy.txt | 0.245406 |

The system prompt contains POLICY ASSISTANT, requires answers from supplied context, and requires the fallback for missing information. The user message format is exactly:

```text
CONTEXT:
[source_file.txt]
first retrieved chunk text
[source_file.txt]
second retrieved chunk text
[source_file.txt]
third retrieved chunk text
QUESTION:
the question
```

`answer(question)` returns model text and the sources used. Evaluate all 14 supplied questions; document hits mean expected_doc occurs among the top three, facts mean a case-sensitive expected_fact substring, and uncovered answers must start `I could not find`.

| Chunk overlap | Total chunks | Expected doc / 12 | Expected fact / 12 | Uncovered fallback / 2 |
| --- | --- | --- | --- | --- |
| 10 | 15 | 12 | 12 | 2 |
| 0 | 13 | 12 | 10 | 2 |

Questions failing grounding with overlap zero:

| Question | Expected fact | Why it fails |
| --- | --- | --- |
| What delivery time makes a completed order eligible for a full refund? | more than 45 minutes | The condition more than 45 minutes is in words 0-39, while the full-refund result starts at word 40. The mock selects the incomplete result sentence; the 30-69 chunk keeps the condition and result together. |
| How soon is money returned when a payment fails but is deducted? | 48 hours | The failed-payment condition ends in words 0-39 and within 48 hours begins at word 40. The zero-overlap retrieved context omits the timing chunk; overlapping words 30-69 retain both. |

### Tool definitions and refund rules

`part_d/tools.py` holds TOOLS_SPEC with one-sentence descriptions and JSON parameter schemas plus TOOL_MAP mapping names to functions. Model requests resolve by their name and arguments dictionary.

| Tool | Parameters | Behavior |
| --- | --- | --- |
| get_order | order_id | Read an order's status, amount and delivery minutes from SQLite. |
| search_policy | query | Search three policy chunks using TF-IDF cosine similarity. |
| calculate_refund | order_amount, delivery_minutes | Calculate a full refund for delivery over 45 minutes or zero otherwise. |
| issue_refund | order_id, refund_inr | Record an eligible completed-order refund no greater than INR 1000. |
| escalate_to_human | order_id, refund_inr | Record an eligible refund above INR 1000 for human approval. |

`get_order` uses a parameterized read-only SQLite query and returns order_id, status, order_amount, delivery_minutes or {"error":"order not found"}. `search_policy` wraps the same retrieve output under results. `calculate_refund` returns refund_inr: full amount only when duration exceeds 45 minutes, otherwise zero. Dispatch enforces each tool's declared required fields, types and numeric bounds and rejects undeclared fields, including database or ledger path overrides. Action tools independently check order existence, completed status, eligible amount and the INR 1,000 threshold before appending a timestamped local ledger action. Values at 45 minutes yield zero; 45.01 minutes qualifies. An escalation records a human approval request and does not issue money. These are simulated ledger entries, not real payment transfers. As required, every supplied request is assumed within seven days.

### ReAct scenarios and measured calls

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with AGENT and the user message. It appends each assistant tool request, dispatches through TOOL_MAP, appends the JSON tool observation and continues. The final prefix supplies the normal status. Runs return status, final_answer, steps, llm_calls and tokens_used, and retain every tool name, arguments and observation in JSONL.

| Scenario | Status | Refund/request amount (INR) | Model calls | Estimated tokens |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 1077 |
| S2 | REFUND | 135.00 | 5 | 1077 |
| S3 | ESCALATE | 1,999.00 | 5 | 1064 |
| S4 | DENIED | 0.00 | 4 | 749 |
| S5 | DENIED | 0.00 | 3 | 445 |
| S6 | NOT_FOUND | 0.00 | 2 | 144 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3216 |

`outputs/agent_runs.jsonl` contains exactly seven current batch lines. Demo runs live in `outputs/agent_demos.jsonl`. S1 issues INR 128.25, S2 issues INR 135.00, and S3 requests human approval for INR 1,999.00. S4 is an on-time denial; S5 is a cancelled-order denial; S6 is not found. S7 completes six calls under the default token limit and hands off by max iterations.

### Independent safety stops

At the top of each iteration, compare elapsed time from time.monotonic against the timeout; elapsed >= the limit stops before another call. Sum usage.total_tokens after every model response; if the total exceeds token_limit, stop before executing the just-requested tool. After the maximum model calls without a final answer, hand off. Every fallback returns completed steps and a short human-handoff summary.

| Demonstration | Status | Model calls | Tokens | Tools completed |
| --- | --- | --- | --- | --- |
| S1 token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 462 | get_order, search_policy |
| time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | None |
| S7 max_iterations=6 | FALLBACK_MAX_ITERATIONS | 6 | 3216 | Six policy searches |

### Crash, checkpoint and resume

After every tool observation, atomically save run id, scenario id, full messages, cumulative tokens/calls, step count, executed tools, pending calls, elapsed active time and limits in `state/<run_id>.json`. The S1 demo raises a simulated crash immediately after saving its second observation, then calls `resume_agent(run_id)`.

| Phase | Executed tools |
| --- | --- |
| Before crash | get_order, search_policy |
| After resume | calculate_refund, issue_refund |
| Full ordered sequence | get_order, search_policy, calculate_refund, issue_refund |

Exactly 4 tool executions occur across both phases, with no repeats; final status is REFUND for INR 128.25. Model calls are 5 cumulatively, including 3 after resume. A terminal resume returns the same result and adds no action. The ledger uses a stable run/step operation id to suppress a replayed local action.

### ReAct versus plan-and-execute (under 250 words)

The measured ReAct workflow uses 5 model calls and 1077 estimated tokens for S1, 3 calls/445 tokens for the cancelled order S5, and 2 calls/144 tokens for NOT_FOUND S6. Each new call resends accumulated history, so its token cost grows with the number and size of observations. A plan-and-execute design could request the full conditional step list once, then execute validated tools locally and replan only on an error; a normal request could use one planning call plus an optional final response call. Its exact token saving is unmeasured because the supplied mock implements ReAct, not a planner. ReAct adapts immediately to NOT_FOUND or a cancelled status and stops before money is issued. A static plan must branch on those observations or replan; blindly following the remaining steps is unsafe. For this fixed refund workflow I would choose a validated conditional plan executed by deterministic Python rules, with human escalation above INR 1000. It reduces repeated model calls while retaining action-time checks. ReAct is preferable when policy or observations need more open-ended reasoning.

## Limitations and Next Steps

The synthetic data and deterministic mock are learning fixtures; performance cannot be treated as real Instamart behavior or production readiness. The perfect few-shot accuracy and exact retrieval facts follow the supplied templates, and answer snippets can be incomplete or contain overlapping wording. Real tickets and policy questions need separate retrieval, grounding and routing evaluation.

The Premium flag is a post-hoc whole-window RFM label that includes September; therefore the retention list is illustrative and is not a leakage-free evaluation. The model's order features remain strictly pre-cutoff, but city/category delivery medians in the mandated Part A cleaner use the whole order window. For a fully historical deployment evaluation, fit imputation statistics on data available before each cutoff as well, then use time-based validation and fit customer segments at campaign time.

The test set is small and customer-level rather than repeated rolling periods. Tune campaign thresholds on separate validation data, measure retention lift and offer costs, and compare performance across cities and customer groups. Oversampling raises recall at the cost of precision; feature importance and coefficients describe associations, not causation.

The Part C budget intentionally gates usage already consumed, allowing a final response to cross the threshold. A hard production spend cap would require reserving a response allowance before a call. The project masks the three prescribed identifiers, but a real privacy pipeline would also cover addresses and other identifiers and protect checkpoint/log access.

Refund actions are local JSONL simulations. All request dates are assumed within seven days, as instructed. The local run/step key handles replay for the single-process recovery demo; concurrent workers and real payment systems would need transactions, locking, provider idempotency keys and order-level duplicate-refund protection. Persisted resume time counts active processing time, not downtime between processes. Ledger timestamps and checkpoint elapsed times vary between runs.

The supplied files remain byte-for-byte unmodified, and verification checks their SHA-256 hashes. All solver code uses the permitted standard library, pandas, numpy, scikit-learn and joblib; requirements include their frozen transitive dependencies. Optional Jupyter packages serve only interactive notebook editing. Review and understand the notebooks before submission, and follow your programme's academic-integrity and assistance-disclosure requirements. Commit the text/code, clean database, required model/CSV/JSONL outputs and checkpoints to `main`, then push them to the public repository; keep credentials and virtual environments excluded.
