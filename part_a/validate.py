"""part_a/validate.py - Validation script for Part A checkpoints."""
import os
import pandas as pd
import numpy as np

def validate_part_a(clean_dir=os.path.join("data", "clean")):
    cust_path = os.path.join(clean_dir, "customers.csv")
    prod_path = os.path.join(clean_dir, "products.csv")
    ord_path = os.path.join(clean_dir, "orders.csv")

    assert os.path.exists(cust_path), f"Missing {cust_path}"
    assert os.path.exists(prod_path), f"Missing {prod_path}"
    assert os.path.exists(ord_path), f"Missing {ord_path}"

    cust = pd.read_csv(cust_path)
    prod = pd.read_csv(prod_path)
    orders = pd.read_csv(ord_path)

    # 1. City counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184
    city_counts = cust["city"].value_counts().to_dict()
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }
    assert city_counts == expected_cities, f"City counts mismatch: {city_counts}"

    # 2. Gender counts: female 610, male 560, unknown 30
    gender_counts = cust["gender"].value_counts().to_dict()
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert gender_counts == expected_genders, f"Gender counts mismatch: {gender_counts}"

    # 3. Missing ages: 0
    assert cust["age"].isna().sum() == 0, "Found missing ages in customers"

    # 4. Every age between 18 and 60
    assert cust["age"].min() >= 18 and cust["age"].max() <= 60, f"Ages outside 18-60: [{cust['age'].min()}, {cust['age'].max()}]"

    # 5. Total clean orders rows: 12,309 (one per order_id)
    assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"
    assert orders["order_id"].nunique() == 12309, f"Order IDs not unique in orders table"

    # 6. Completed orders: 11,110
    completed_orders = orders[orders["status"] == "completed"]
    assert len(completed_orders) == 11110, f"Expected 11110 completed orders, got {len(completed_orders)}"

    # 7. Cancelled orders: 1,199
    cancelled_orders = orders[orders["status"] == "cancelled"]
    assert len(cancelled_orders) == 1199, f"Expected 1199 cancelled orders, got {len(cancelled_orders)}"

    # 8. Pending orders: 0
    assert (orders["status"] == "pending").sum() == 0, "Found pending orders in clean table"

    # 9. 0 missing order_amount
    assert orders["order_amount"].isna().sum() == 0, "Found missing order_amount"

    # 10. 0 missing delivery_minutes among completed orders
    assert completed_orders["delivery_minutes"].isna().sum() == 0, "Found missing delivery_minutes in completed orders"

    # 11. 1,199 missing delivery_minutes among cancelled orders
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, f"Expected 1199 missing deliv in cancelled, got {cancelled_orders['delivery_minutes'].isna().sum()}"

    # 12. Sum of order_amount over completed orders is INR 2,998,668.80
    completed_sum = round(completed_orders["order_amount"].sum(), 2)
    assert abs(completed_sum - 2998668.80) < 0.01, f"Expected sum 2998668.80, got {completed_sum}"

    print("ALL CHECKPOINTS PASSED")
    return True

if __name__ == "__main__":
    validate_part_a()
