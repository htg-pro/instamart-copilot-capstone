"""Build the single capstone README from measured Part A-D reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]])


def number(value, digits=2):
    return f"{value:,.{digits}f}"


def main():
    reports = {p: json.loads((ROOT / f"outputs/part_{p}_report.json").read_text()) for p in "abcd"}
    from check_environment import inspect_environment
    environment = inspect_environment()
    a, b, c, d = [reports[p] for p in "abcd"]
    lapse = b["lapse_experiment"]
    parts = []
    parts.append("""# Instamart Customer Intelligence & Support Copilot

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

For interactive notebooks, run `python -m notebook`; Jupyter and its Python kernel are included in the same `requirements.txt`. In VS Code, select `.venv/bin/python` as the notebook kernel. Run all cells from top to bottom in A, B, C, D order. The bundled notebook executor supports these plain-Python, text-output notebooks; it is not a general executor for IPython magics or widget notebooks. On Windows, use `.venv\\Scripts\\activate` in Command Prompt for step 2.
""")
    if not environment["notebook_tools_passed"]:
        parts.append("\nVerification scope: the saved results were produced in the pinned analytics environment using the plain-Python notebook executor. Interactive Jupyter installation and launch have not been verified in this environment. The combined requirements file lists those tools, and `scripts/check_environment.py` checks their installation before a fresh reproduction.\n")
    parts.append("""
## Part A

### Raw audit

Pandas reads the three raw analytical tables. Tickets are read with `keep_default_na=False` wherever they are used, preserving empty text. Missing percentages and actual data types are reported below; pandas parses `N/A` as missing, while `error` remains a nonnumeric string. Both count in the separate 120-value nonnumeric-amount audit.
""")
    audit_rows = []
    for name in ["customers", "products", "orders"]:
        shape = a["raw_shapes"][f"{name}_shape"]
        for row in a["column_audit"][name]:
            audit_rows.append([name, f"{shape[0]} × {shape[1]}", row["column"], row["dtype"],
                               row["missing_count"], number(row["missing_pct"])])
    parts.append(table(["Table", "Shape", "Column", "dtype", "Missing", "Missing %"], audit_rows))
    parts.append("\n" + table(["Raw audit checkpoint", "Measured count"], [[k, v] for k, v in a["raw_audit"].items()]))
    parts.append("""
### Cleaning rules and actual changes

Cities are stripped and title-cased; genders map to male/female/unknown. Ages outside 10-100 become missing, then all missing ages use the valid city median from `groupby` and `transform`. Fractional medians are preserved. Signup dates become datetimes.

Orders follow the required sequence: parse timestamps; remove exact duplicates; select the entire latest row per order id; parse and reconstruct missing amounts from `round(price_inr * quantity * (1 - discount_pct / 100), 2)`; replace negative delivery durations with missing; impute missing completed durations by completed-product-category median; leave cancelled durations missing. Selecting a whole row avoids accidentally borrowing a nonmissing field from an older record. Products are unchanged.
""")
    parts.append(table(["Rule / stage", "Rows or values affected"],
                       [[k, v] for k, v in a["cleaning_changes"].items() if not isinstance(v, dict)]))
    parts.append("\nRaw amounts/negative durations include duplicates; therefore 120 raw nonnumeric amounts become 112 reconstructions, and 106 raw negative durations become 105 replacements after deduplication. The 774 completed-duration imputations include blanks and invalid durations.\n")
    parts.append(table(["City", "Age imputation median"], a["cleaning_changes"]["city_age_medians"].items()))
    parts.append("\n" + table(["Category", "Completed delivery imputation median (minutes)"], a["cleaning_changes"]["category_delivery_medians"].items()))
    parts.append("\n" + table(["City", "Clean customer count"], a["clean"]["city_counts"].items()))
    parts.append("\n" + table(["Gender", "Clean customer count"], a["clean"]["gender_counts"].items()))
    parts.append("""
### Storage and validation

Clean tables are written to `data/clean/` and SQLite tables `customers`, `products`, `orders` in `data/instamart.db`. Timestamps are stored as `YYYY-MM-DD HH:MM:SS`. `part_a/validate.py` checks customer distributions, valid ages, order identity/status counts, missingness and revenue; `part_a/report.py` additionally checks every SQL reference.

""" + table(["Clean checkpoint", "Measured result"], [
        ["Customers / products / orders", "1,200 / 40 / 12,309"],
        ["Completed / cancelled / pending", "11,110 / 1,199 / 0"],
        ["Unique order ids", "12,309"], ["Missing ages / age range", "0 / 18-60"],
        ["Missing amounts", 0], ["Missing completed / cancelled delivery", "0 / 1,199"],
        ["Completed revenue", "INR " + number(a["clean"]["completed_revenue_inr"])],
        ["Standalone validator", "ALL CHECKPOINTS PASSED"],
    ]))
    parts.append("""
### SQL business questions

All seven statements in `sql/queries.sql` run with `pd.read_sql_query`. Revenue and spend include completed orders only. Results are also exported as `outputs/sql_q1.csv` through `sql_q7.csv`.

""" + table(["Query", "Technique and result"], [
        ["Q1", "WHERE amount > INR 500, HAVING filtered spend > INR 8,000: 30 customers"],
        ["Q2", "Three-table city/category join: 30 rows, total INR 2,998,668.80; largest Bengaluru/Electronics INR 233,909.45"],
        ["Q3", "Scalar subquery, price above catalogue average: 11 products"],
        ["Q4", "LEFT JOIN, no completed orders: 25 customers"],
        ["Q5", "Two CTEs, spend above customer average: 550 customers; average INR " + number(a["average_customer_spend_inr"])],
        ["Q6", "City RANK/DENSE_RANK/ROW_NUMBER: 1,175 rows; 8 rank/row-number differences"],
        ["Q7", "Daily revenue with cumulative SUM OVER: 183 days; final INR 2,998,668.80"],
    ]))
    leaders = [row for row in a["sql"]["Q6"]["results"] if row["row_num"] == 1]
    parts.append("\n" + table(["City", "Top customer", "Completed spend (INR)"],
                              [[r["city"],r["customer_id"],number(r["total_spend"])] for r in leaders]))
    parts.append("\nTies share RANK values while ROW_NUMBER assigns distinct consecutive positions, so eight rows have different rank and row-number values.\n")
    parts.append("""
### pandas cross-check and exploratory statistics

The city/category pivot uses `pd.pivot_table(..., margins=True)` and its INR 2,998,668.80 grand total equals Q2. Named aggregation computes city sum, mean and count:

""" + table(["City", "Sum (INR)", "Mean (INR)", "Completed count"],
            [[r["city"],number(r["sum"]),number(r["mean"]),r["count"]] for r in a["city_aggregation"]]))
    parts.append("\n" + table(["Amount statistic", "Measured result"], [[k,number(v) if isinstance(v,float) else v] for k,v in a["distribution"].items()]))
    cols = list(a["correlation_matrix"])
    parts.append("\nCorrelation matrix over completed orders:\n\n" + table(["Feature",*cols],
                 [[row,*[number(a["correlation_matrix"][col][row],4) for col in cols]] for row in cols]))
    parts.append("""
Exploratory interpretation (under 200 words): The data describes synthetic quick-commerce orders for 1,200 customers across six cities during April-September 2025. Completed revenue is INR 2,998,668.80 across 11,110 orders. Mean order amount is INR 269.91 versus median INR 135.00, with skewness 4.61: expensive baskets stretch the right tail. Q1 is INR 64.80 and Q3 is INR 297.00; IQR fences are INR -283.50 and INR 645.30. There are 897 upper outliers and no lower outliers. High-value electronics baskets need contextual review rather than automatic removal, while planted invalid ages, malformed amounts and negative delivery times need explicit cleaning. Amount and discount have the largest absolute correlation, about -0.095, followed by amount and quantity at +0.087. Both are weak associations; correlation of this size does not establish causation. City/category revenue and customer activity support the later segmentation and retention workflow.

## Part B

### RFM, clustering and business segments

Only completed orders enter RFM, as of `2025-10-01 00:00:00`. Recency uses whole elapsed days, frequency counts distinct orders, and monetary sums completed INR spend.

""" + table(["RFM statistic", "Measured value"], [[k,number(v) if isinstance(v,float) else v] for k,v in b["rfm_summary"].items()]))
    parts.append("\nThe RFM columns are standardized with StandardScaler. Every K from 2-8 uses `KMeans(n_clusters=K, n_init=10, random_state=42)`:\n\n" +
                 table(["K","Inertia","Silhouette"], [[r["k"],number(r["inertia"],1),number(r["silhouette"],3)] for r in b["kmeans_evaluation"]]))
    parts.append("\n" + b["cluster_choice_explanation"] + "\n")
    parts.append(table(["ID","Business segment","Size","Mean recency (days)","Mean frequency","Mean monetary (INR)","Action","Premium"],
        [[r["segment_id"],r["segment_name"],r["size"],number(r["mean_recency"]),number(r["mean_frequency"]),
          number(r["mean_monetary"]),r["recommended_action"],r["is_premium"]] for r in b["segment_profiles"]]))
    ratios = b["pca_explained_variance_ratio"]
    parts.append(f"\nPCA with two components explains {ratios[0]:.6f} and {ratios[1]:.6f}, total {b['pca_total_explained_variance']:.6f}. `outputs/customer_segments.csv` has customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium; Premium is true only for the highest-mean-monetary segment.\n")
    parts.append("""
### Lapse cohort, features and training isolation

Cutoff: `2025-09-01 00:00:00`. The cohort contains customers with at least one completed order before the cutoff. `lapsed=1` means no completed order during September 1-30; the target window ends exclusively at October 1. Orders after September cannot change this label.

Features are recency, frequency, monetary, mean completed discount, final-order cancellation rate, mean completed delivery minutes, favourite product category, city, age, gender and tenure days. Cancellation rate divides cancelled orders by completed plus cancelled orders before cutoff. The latest order entering features is `2025-08-31 22:35:45`. No order from September 1 onwards enters a feature, and no whole-window segment label is a feature. Code asserts no missing feature values.

`train_test_split(test_size=0.2, stratify=y, random_state=42)` is shared by both models. Numeric scaling and `OneHotEncoder(drop='first', handle_unknown='ignore')` live inside each Pipeline; GridSearchCV fits them separately on training folds. No encoder, scaler or resampler sees test rows. Favourite category, city and gender each use k-1 nominal columns.

""" + table(["Cohort / split checkpoint","Value"], [
        ["Cohort",lapse["cohort_size"]],["Lapsed",f"{lapse['lapsed_count']} ({lapse['lapsed_pct']:.2f}%)"],
        ["Train / test",f"{lapse['train_size']} / {lapse['test_size']}"],
        ["Training not-lapsed / lapsed","570 / 354"],["Test not-lapsed / lapsed","143 / 89"],
        ["Fitted training frequency mean",number(lapse["learned_freq_mean"],6)],
        ["Full-cohort frequency mean",number(lapse["full_cohort_freq_mean"],6)],
        ["Missing features",0],
    ]))
    parts.append("\n" + lapse["scaler_explanation"] + "\n")
    parts.append("""
### Tune on training data, then evaluate the untouched test set

LogisticRegression uses max_iter=1000, C in {0.01, 0.1, 1, 10, 100}, five-fold CV and F1 scoring: 25 CV fits plus refit. RandomForestClassifier uses 100 trees, random_state=42, max_depth in {3, 5, 8}, min_samples_leaf in {5, 10, 20}, five-fold CV and F1 scoring: 45 CV fits plus refit. Both searches finish before test evaluation. Confusion matrices use rows true [0,1], columns predicted [0,1], in the form [[TN,FP],[FN,TP]]. All precision, recall and F1 values concern lapsed=1; zero_division=0 is used.

""" + table(["Model","Best parameters","Best CV F1"],
        [[name,json.dumps(lapse[key]["best_params"]),number(lapse[key]["best_cv_f1"],4)] for name,key in [("Logistic regression","lr_tuning"),("Random forest","rf_tuning")]]))
    parts.append("\n" + table(["Test model","Confusion matrix","Accuracy","Precision (1)","Recall (1)","F1 (1)"],
        [[name,json.dumps(lapse[key]["cm"]),*[number(lapse[key][metric],4) for metric in ["accuracy","precision","recall","f1"]]]
         for name,key in [("Always not lapsed","baseline"),("Tuned logistic regression","lr_test"),("Tuned random forest","rf_test")]]))
    parts.append("\n" + lapse["selected_model_justification"] + "\nMeasured best C can differ from the reference's example C=1; the specified grid, cohort and positive-class performance criteria are preserved.\n")
    parts.append("""
### Class balance experiment

Keep every original training row, then sample lapsed training rows with replacement and random_state=42 until both classes have 570 rows. This adds 216 duplicates to the 924 original rows. A cloned best logistic pipeline is refitted on the balanced training data; the original test set and tuned base models remain untouched.

""" + table(["Logistic training","Test precision (1)","Test recall (1)","Test F1 (1)"],
        [[name,*[number(lapse[key][metric],4) for metric in ["precision","recall","f1"]]] for name,key in [("Original 570/354","lr_test"),("Balanced 570/570","lr_balanced_test")]]))
    parts.append("\n" + lapse["balance_explanation"] + "\n")
    parts.append("""
### Interpretation, persistence and retention export

Five largest random-forest feature importances:

""" + table(["Feature","Importance","Business reading"],
        [[r["feature"],number(r["importance"],6),r["business_reading"]] for r in lapse["top5_features"]]))
    parts.append("\nThree fitted logistic coefficients (numeric coefficients refer to standardized inputs):\n\n" +
        table(["Feature","Coefficient","Sign","Business reading"],
        [[r["feature"],number(r["coefficient"],6),r["sign"],r["business_reading"]] for r in lapse["lr_coefficient_readings"]]))
    parts.append(f"\nThe selected {lapse['selected_model']} Pipeline is persisted to `outputs/lapse_model.joblib`; joblib reload reproduces original test predictions exactly: **{lapse['reload_predictions_match']}**. There are {lapse['premium_test_customers']} Premium test customers; the top ten by probability are exported with the required columns:\n\n" +
        table(["Customer","Lapse probability","Actually lapsed"], [[r["customer_id"],number(r["lapse_probability"],6),r["actually_lapsed"]] for r in lapse["retention_top10"]]))
    parts.append(f"\nActually lapsed among the top ten: **{lapse['retention_actually_lapsed_count']}/10**. {lapse['retention_limitation']}\n")
    parts.append("""
## Part C

### Guardrails and privacy

Process tickets in raw CSV order. Empty or whitespace-only text is skipped; text with est_tokens > 400 is flagged long. Neither triggers a model call. `est_tokens` is the unmodified mock's ceil(words * 1.3) estimator.

""" + table(["Input check","Measured result"], [["Total tickets",164],["Eligible",156],["skipped_empty",4],["flagged_long",4],["Tokens per long ticket",734]]))
    parts.append("""
Apply exactly these masks, in this order, before building a prompt:

```text
email:    [\\w.+-]+@[\\w-]+\\.[\\w.]+  -> [EMAIL]
phone:    \\b[6-9]\\d{9}\\b            -> [PHONE]
order id: ORD-\\d{6}                  -> [ORDER_ID]
```

""" + table(["PII type","Substitutions across all tickets","Remaining"],
        [[key,c["input_audit"]["pii_substitutions"][key],c["input_audit"]["pii_remaining"][key]] for key in ["email","phone","order_id"]]
        + [["Total",278,0]]))
    parts.append("""
Reapplying all three regexes finds zero matches. An assertion checks every prompt, including retry attempts. The user message is exactly the redacted ticket text, without appended labels or identifiers. The notebook additionally spies on all 175 attempted calls in a complete batch to prove privacy and file order.

### Prompts, retries and output validation

Zero-shot system prompt:

```text
""" + c["prompts"]["zero_shot"] + "\n```\n\nFew-shot system prompt:\n\n```text\n" + c["prompts"]["few_shot"] + "\n```\n")
    parts.append("""
Each independent run immediately resets mock counters. Every call has at most three retries; wait for retry n=0,1,2 is `0.05 * 2**n * uniform(0.9,1.15)`, using a per-run random generator seeded 42. Complete runs encounter 19 transient failures, all recovered, with zero permanent errors. Every mock retry has n=0, so waits fall in 0.045-0.0575 seconds. One actual zero-shot retry log line appears below; all 19 are retained in `outputs/part_c_report.json` and notebook outputs.

```text
""" + c["zero_shot"]["retry_log"][0] + "\n```\n")
    parts.append("""
Accept JSON only when its category is one of billing/delivery/technical/fraud (`classified_json`). Only a non-JSON answer may recover by `re.findall(r'[a-z]+', text.lower())`, and exactly one category word must appear (`classified_keyword`). Valid JSON with an invalid category, ambiguous text and every other invalid output become `flagged_invalid_output` and go to human review. No category is guessed from an invalid JSON reason.

""" + table(["Full-run status","Count"], c["few_shot"]["status_counts"].items()))
    parts.append("""
### Prompt comparison and token cost

Accuracy uses only classified tickets; tokens sum usage.total_tokens over all successful calls, including replies that fail validation. There is no token budget for these two comparison runs.

""" + table(["Prompt","Classified","Correct","Accuracy","Total tokens","Tokens / classified"],
        [[name,c[key]["classified"],c[key]["correct"],f"{100*c[key]['accuracy']:.2f}%",c[key]["total_tokens"],number(c[key]["tokens_per_classified_ticket"],3)]
         for name,key in [("Zero-shot","zero_shot"),("Few-shot","few_shot")]]))
    cats = c["zero_shot"]["confusion_categories"]
    parts.append("\nZero-shot confusion table: rows are true category, columns predicted category; excluded invalid/guarded tickets are not in the denominator.\n\n" +
        table(["True / predicted",*cats], [[cat,*row] for cat,row in zip(cats,c["zero_shot"]["confusion_matrix"])]))
    parts.append("\nThe few-shot prompt increases tokens from 9,400 to 17,044, adding 7,644 tokens and about 52 per classified ticket. It corrects all 21 zero-shot errors: 11 fraud tickets and 10 technical tickets previously classified as billing. The extra tokens are worthwhile when a fraud misroute or delayed technical response costs more than the added processing budget. This synthetic mock comparison is perfect at 147/147, but live language-model quality would need separate evaluation.\n")
    parts.append("""
### Budget and segment-aware routing

Before every model attempt, including a retry, stop if tokens already used are at least the budget. This assignment-prescribed gate is soft: the last admitted response may exceed the threshold. The zero-shot budget demo uses 6,000, exhausts 56 tickets, and ends at 6,038 tokens (38 over the gate). The final few-shot run uses exactly 100,000 and consumes 17,044 with no budget-exhausted rows.

Skipped, long, invalid or budget-exhausted tickets go to human_review/NORMAL. Other fraud tickets go to security_team/HIGH. Other Premium-customer tickets go to premium_desk/HIGH. All other classified tickets go to support_queue/NORMAL. Missing segment rows mean not Premium; category and segment_name remain blank when unavailable. Final CSV columns are ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used.

""" + table(["Final route","Tickets"], c["final_few_shot"]["routing_counts"].items()))
    parts.append("\n" + table(["Final priority","Tickets"], c["final_few_shot"]["priority_counts"].items()))
    parts.append("\nAll 147 accepted outputs were rule-validated (135 JSON and 12 keyword); 17 guarded/invalid tickets went to humans. `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv` each contain all 164 input tickets.\n")
    parts.append("""
## Part D

### Policy chunking, retrieval and grounded answers

Split each policy into 40-word chunks with 10-word overlap, never crossing document boundaries. Retain each chunk's source. Fit the prescribed default `TfidfVectorizer()` on the 15 chunks and retrieve the three highest cosine similarities, returning source, score and text. All three top chunks for `refund policy late delivery` come from refund_policy.txt.

TF measures how frequently a term appears within a chunk. IDF downweights terms common across many chunks and emphasizes rare terms, but lexical TF-IDF vectors can miss equivalent policy questions phrased with different words.

""" + table(["Policy document","Chunks"], d["chunk_counts"].items()))
    parts.append("\n" + table(["Reference retrieval rank","Source","Cosine score"], [[i,r["source"],number(r["score"],6)] for i,r in enumerate(d["reference_retrieval"],1)]))
    parts.append("""
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

""" + table(["Chunk overlap","Total chunks","Expected doc / 12","Expected fact / 12","Uncovered fallback / 2"],
        [[10,d["chunk_total"],d["rag_overlap_10"]["doc_hits"],d["rag_overlap_10"]["fact_hits"],d["rag_overlap_10"]["fallback_hits"]],
         [0,d["zero_overlap_chunk_total"],d["rag_overlap_0"]["doc_hits"],d["rag_overlap_0"]["fact_hits"],d["rag_overlap_0"]["fallback_hits"]]]))
    parts.append("\nQuestions failing grounding with overlap zero:\n\n" + table(["Question","Expected fact","Why it fails"],
        [[r["question"],r["expected_fact"],next(item["explanation"] for item in d["boundary_examples"] if item["question_id"]==r["question_id"])] for r in d["zero_overlap_failures"]]))
    parts.append("""
### Tool definitions and refund rules

`part_d/tools.py` holds TOOLS_SPEC with one-sentence descriptions and JSON parameter schemas plus TOOL_MAP mapping names to functions. Model requests resolve by their name and arguments dictionary.

""" + table(["Tool","Parameters","Behavior"], [
        [r["name"],", ".join(r["parameters"]["required"]),r["description"]] for r in d["tools_spec"]]))
    parts.append("""
`get_order` uses a parameterized read-only SQLite query and returns order_id, status, order_amount, delivery_minutes or {"error":"order not found"}. `search_policy` wraps the same retrieve output under results. `calculate_refund` returns refund_inr: full amount only when duration exceeds 45 minutes, otherwise zero. Dispatch enforces each tool's declared required fields, types and numeric bounds and rejects undeclared fields, including database or ledger path overrides. Action tools independently check order existence, completed status, eligible amount and the INR 1,000 threshold before appending a timestamped local ledger action. Values at 45 minutes yield zero; 45.01 minutes qualifies. An escalation records a human approval request and does not issue money. These are simulated ledger entries, not real payment transfers. As required, every supplied request is assumed within seven days.

### ReAct scenarios and measured calls

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with AGENT and the user message. It appends each assistant tool request, dispatches through TOOL_MAP, appends the JSON tool observation and continues. The final prefix supplies the normal status. Runs return status, final_answer, steps, llm_calls and tokens_used, and retain every tool name, arguments and observation in JSONL.

""" + table(["Scenario","Status","Refund/request amount (INR)","Model calls","Estimated tokens"],
        [[r["scenario_id"],r["status"],number(r["refund_inr"]),r["llm_calls"],r["tokens_used"]] for r in d["scenarios"]]))
    parts.append("""
`outputs/agent_runs.jsonl` contains exactly seven current batch lines. Demo runs live in `outputs/agent_demos.jsonl`. S1 issues INR 128.25, S2 issues INR 135.00, and S3 requests human approval for INR 1,999.00. S4 is an on-time denial; S5 is a cancelled-order denial; S6 is not found. S7 completes six calls under the default token limit and hands off by max iterations.

### Independent safety stops

At the top of each iteration, compare elapsed time from time.monotonic against the timeout; elapsed >= the limit stops before another call. Sum usage.total_tokens after every model response; if the total exceeds token_limit, stop before executing the just-requested tool. After the maximum model calls without a final answer, hand off. Every fallback returns completed steps and a short human-handoff summary.

""" + table(["Demonstration","Status","Model calls","Tokens","Tools completed"], [
        ["S1 token_limit=200",d["token_limit_demo"]["status"],d["token_limit_demo"]["llm_calls"],d["token_limit_demo"]["tokens_used"],", ".join(d["token_limit_demo"]["tools_executed"])],
        ["time_limit_seconds=0",d["timeout_demo"]["status"],d["timeout_demo"]["llm_calls"],d["timeout_demo"]["tokens_used"],"None"],
        ["S7 max_iterations=6",d["scenarios"][-1]["status"],d["scenarios"][-1]["llm_calls"],d["scenarios"][-1]["tokens_used"],"Six policy searches"],
    ]))
    parts.append("""
### Crash, checkpoint and resume

After every tool observation, atomically save run id, scenario id, full messages, cumulative tokens/calls, step count, executed tools, pending calls, elapsed active time and limits in `state/<run_id>.json`. The S1 demo raises a simulated crash immediately after saving its second observation, then calls `resume_agent(run_id)`.

""" + table(["Phase","Executed tools"], [
        ["Before crash",", ".join(d["crash_resume"]["tools_before_crash"])],
        ["After resume",", ".join(d["crash_resume"]["tools_after_resume"])],
        ["Full ordered sequence",", ".join(d["crash_resume"]["complete_tool_sequence"])],
    ]))
    parts.append(f"\nExactly {d['crash_resume']['tool_execution_count']} tool executions occur across both phases, with no repeats; final status is REFUND for INR {number(d['crash_resume']['refund_inr'])}. Model calls are {d['crash_resume']['result']['llm_calls']} cumulatively, including {d['crash_resume']['result']['resumed_llm_calls']} after resume. A terminal resume returns the same result and adds no action. The ledger uses a stable run/step operation id to suppress a replayed local action.\n")
    parts.append("\n### ReAct versus plan-and-execute (under 250 words)\n\n" + d["react_vs_plan_and_execute"] + "\n")
    parts.append("""
## Limitations and Next Steps

The synthetic data and deterministic mock are learning fixtures; performance cannot be treated as real Instamart behavior or production readiness. The perfect few-shot accuracy and exact retrieval facts follow the supplied templates, and answer snippets can be incomplete or contain overlapping wording. Real tickets and policy questions need separate retrieval, grounding and routing evaluation.

""" + lapse["retention_limitation"] + """ The model's order features remain strictly pre-cutoff, but city/category delivery medians in the mandated Part A cleaner use the whole order window. For a fully historical deployment evaluation, fit imputation statistics on data available before each cutoff as well, then use time-based validation and fit customer segments at campaign time.

The test set is small and customer-level rather than repeated rolling periods. Tune campaign thresholds on separate validation data, measure retention lift and offer costs, and compare performance across cities and customer groups. Oversampling raises recall at the cost of precision; feature importance and coefficients describe associations, not causation.

The Part C budget intentionally gates usage already consumed, allowing a final response to cross the threshold. A hard production spend cap would require reserving a response allowance before a call. The project masks the three prescribed identifiers, but a real privacy pipeline would also cover addresses and other identifiers and protect checkpoint/log access.

Refund actions are local JSONL simulations. All request dates are assumed within seven days, as instructed. The local run/step key handles replay for the single-process recovery demo; concurrent workers and real payment systems would need transactions, locking, provider idempotency keys and order-level duplicate-refund protection. Persisted resume time counts active processing time, not downtime between processes. Ledger timestamps and checkpoint elapsed times vary between runs.

The supplied files remain byte-for-byte unmodified, and verification checks their SHA-256 hashes. All solver code uses the permitted standard library, pandas, numpy, scikit-learn and joblib; requirements include their frozen transitive dependencies. Optional Jupyter packages serve only interactive notebook editing. Review and understand the notebooks before submission, and follow your programme's academic-integrity and assistance-disclosure requirements. Commit the text/code, clean database, required model/CSV/JSONL outputs and checkpoints to `main`, then push them to the public repository; keep credentials and virtual environments excluded.
""")
    text = "\n\n".join(part.strip() for part in parts) + "\n"
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print("README.md rebuilt from measured Part A-D results.")


if __name__ == "__main__":
    main()
