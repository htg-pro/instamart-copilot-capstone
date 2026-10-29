"""part_a/run_part_a.py - End-to-end execution of Part A."""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_a.audit import audit_raw_data
from part_a.clean import run_cleaning
from part_a.db import store_to_sqlite, run_sql_queries
from part_a.validate import validate_part_a
from part_a.eda import run_eda

def main():
    print("==================================================")
    print("           PART A: DATA FOUNDATION PIPELINE       ")
    print("==================================================")

    # 1. Audit
    print("\n--- Step 1: Auditing Raw Data ---")
    summary, audit_metrics, col_summaries = audit_raw_data()
    print(f"Raw Order Rows: {audit_metrics['raw_order_rows']}")
    print(f"Exact Duplicate Rows: {audit_metrics['exact_duplicates']}")
    print(f"Pending Status Rows: {audit_metrics['pending_rows']}")
    print(f"Distinct order_id: {audit_metrics['distinct_order_ids']}")
    print(f"Non-numeric order_amount: {audit_metrics['non_numeric_amounts']}")
    print(f"Negative delivery_minutes: {audit_metrics['negative_delivery_times']}")
    print(f"Blank delivery_minutes: {audit_metrics['blank_delivery_times']}")
    print(f"Blank age in customers: {audit_metrics['blank_ages']}")
    print(f"Ages outside 10-100: {audit_metrics['ages_outside_range']}")
    print(f"Distinct raw cities: {audit_metrics['distinct_raw_cities']}")
    print(f"Distinct raw genders (non-blank): {audit_metrics['distinct_raw_genders']}")
    print(f"Blank genders: {audit_metrics['blank_genders']}")

    # 2. Clean
    print("\n--- Step 2: Cleaning Data ---")
    clean_cust, clean_prod, clean_ord = run_cleaning()
    print(f"Clean Customers: {clean_cust.shape}")
    print(f"Clean Products: {clean_prod.shape}")
    print(f"Clean Orders: {clean_ord.shape}")

    # 3. Store to SQLite
    print("\n--- Step 3: Storing Clean Tables to SQLite ---")
    db_path = store_to_sqlite(clean_cust, clean_prod, clean_ord)
    print(f"Database created at: {db_path}")

    # 4. Validate
    print("\n--- Step 4: Validating Checkpoints ---")
    validate_part_a()

    # 5. Run SQL Queries
    print("\n--- Step 5: Executing SQL Queries ---")
    sql_results = run_sql_queries()
    for q_name, df in sql_results.items():
        print(f"Query {q_name}: {len(df)} rows returned")

    # 6. EDA and pandas cross-checks
    print("\n--- Step 6: Pandas Cross-check & EDA ---")
    eda_results = run_eda()
    print(f"Pivot Grand Total (Completed Revenue): INR {eda_results['grand_total']:,.2f}")
    print("\nDistribution of Completed order_amount:")
    for k, v in eda_results["distribution_stats"].items():
        print(f"  {k}: {v}")

    print("\nPart A completed successfully!")

if __name__ == "__main__":
    main()
