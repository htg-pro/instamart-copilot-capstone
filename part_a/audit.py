"""part_a/audit.py - Auditing raw data for the Instamart Capstone Project."""
import os
import pandas as pd
import numpy as np

def audit_raw_data(data_dir=os.path.join("data", "raw")):
    customers_path = os.path.join(data_dir, "customers.csv")
    products_path = os.path.join(data_dir, "products.csv")
    orders_path = os.path.join(data_dir, "orders.csv")
    tickets_path = os.path.join(data_dir, "tickets.csv")

    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)
    orders = pd.read_csv(orders_path)
    tickets = pd.read_csv(tickets_path, keep_default_na=False)

    summary = {
        "customers_shape": customers.shape,
        "products_shape": products.shape,
        "orders_shape": orders.shape,
        "tickets_shape": tickets.shape,
    }

    # Specific audit checkpoints
    raw_order_rows = len(orders)
    exact_duplicates = int(orders.duplicated().sum())
    pending_rows = int((orders["status"] == "pending").sum())
    distinct_order_ids = int(orders["order_id"].nunique())

    # Order amount numeric check
    non_numeric_amounts = int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())

    # Delivery minutes checks
    deliv_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_delivery_times = int((deliv_numeric < 0).sum())
    blank_delivery_times = int(orders["delivery_minutes"].isna().sum())

    # Customers checks
    blank_ages = int(customers["age"].isna().sum())
    cust_age_num = pd.to_numeric(customers["age"], errors="coerce")
    ages_outside_range = int(((cust_age_num < 10) | (cust_age_num > 100)).sum())
    distinct_raw_cities = int(customers["city"].nunique())
    distinct_raw_genders = int(customers["gender"].dropna().nunique())
    blank_genders = int(customers["gender"].isna().sum())

    audit_metrics = {
        "raw_order_rows": raw_order_rows,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amounts,
        "negative_delivery_times": negative_delivery_times,
        "blank_delivery_times": blank_delivery_times,
        "blank_ages": blank_ages,
        "ages_outside_range": ages_outside_range,
        "distinct_raw_cities": distinct_raw_cities,
        "distinct_raw_genders": distinct_raw_genders,
        "blank_genders": blank_genders,
    }

    column_summaries = {}
    for name, df in [("customers", customers), ("products", products), ("orders", orders), ("tickets", tickets)]:
        col_info = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = float(missing_count / len(df) * 100.0)
            col_info.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2)
            })
        column_summaries[name] = pd.DataFrame(col_info)

    return summary, audit_metrics, column_summaries

if __name__ == "__main__":
    summary, audit_metrics, col_summaries = audit_raw_data()
    print("=== Raw Data Audit Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n=== Checkpoint Audit Metrics ===")
    for k, v in audit_metrics.items():
        print(f"{k}: {v}")
