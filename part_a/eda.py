"""part_a/eda.py - Pandas cross-check, summary statistics, and exploratory data analysis."""
import os
import pandas as pd
import numpy as np

def run_eda(clean_dir=os.path.join("data", "clean")):
    cust = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    prod = pd.read_csv(os.path.join(clean_dir, "products.csv"))
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))

    completed = orders[orders["status"] == "completed"].copy()
    merged = completed.merge(cust, on="customer_id").merge(prod, on="product_id")

    # (a) Rebuild Q2 with pd.pivot_table
    pivot = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    ).round(2)

    grand_total = float(pivot.loc["Total", "Total"])

    # Per-city named aggregation
    city_agg = (
        merged.groupby("city")["order_amount"]
        .agg(sum="sum", mean="mean", count="count")
        .round(2)
    )

    # (b) Statistical distribution of completed order_amount
    amounts = completed["order_amount"]
    mean_val = float(amounts.mean())
    median_val = float(amounts.median())
    skew_val = float(amounts.skew())
    q1 = float(amounts.quantile(0.25))
    q3 = float(amounts.quantile(0.75))
    iqr = q3 - q1
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)
    lower_outliers = int((amounts < lower_fence).sum())
    upper_outliers = int((amounts > upper_fence).sum())

    stats = {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "skewness": round(skew_val, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_fence": round(lower_fence, 2),
        "upper_fence": round(upper_fence, 2),
        "lower_outliers": lower_outliers,
        "upper_outliers": upper_outliers
    }

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr().round(4)

    return {
        "pivot_table": pivot,
        "grand_total": grand_total,
        "city_aggregation": city_agg,
        "distribution_stats": stats,
        "correlation_matrix": corr_matrix
    }

if __name__ == "__main__":
    res = run_eda()
    print("=== Pivot Table ===")
    print(res["pivot_table"])
    print("\n=== City Aggregation ===")
    print(res["city_aggregation"])
    print("\n=== Distribution Stats ===")
    for k, v in res["distribution_stats"].items():
        print(f"{k}: {v}")
    print("\n=== Correlation Matrix ===")
    print(res["correlation_matrix"])
