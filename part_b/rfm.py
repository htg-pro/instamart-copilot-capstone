"""part_b/rfm.py - Computes RFM metrics for Instamart customers."""
import os
import pandas as pd

def compute_rfm_table(clean_orders_path=os.path.join("data", "clean", "orders.csv"),
                      snapshot_date="2025-10-01 00:00:00"):
    orders = pd.read_csv(clean_orders_path)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Completed orders only
    completed = orders[orders["status"] == "completed"].copy()
    snapshot = pd.Timestamp(snapshot_date)

    rfm = completed.groupby("customer_id").agg(
        recency=("order_ts", lambda ts: int((snapshot - ts.max()).days)),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm

if __name__ == "__main__":
    rfm = compute_rfm_table()
    print("RFM Table Summary:")
    print(f"Total customers: {len(rfm)}")
    print(f"Recency: median={rfm['recency'].median()}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")
