"""part_a/clean.py - Data cleaning pipeline for Instamart datasets."""
import os
import pandas as pd
import numpy as np

def clean_customers_data(raw_customers_df):
    """
    Cleans the raw customers dataframe according to Part A Task 3 rules:
    (a) Standardise city: strip leading/trailing whitespace, apply title case (exactly 6 cities).
    (b) Map gender spellings: male, female, or unknown (blank -> unknown).
    (c) Impute age: ages outside [10, 100] treated as missing, fill missing with city median age.
    (d) Convert signup_date to datetime.
    """
    df = raw_customers_df.copy()

    # Rule (a): Standardise city
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # Rule (b): Map gender
    def map_gender(val):
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        g = str(val).strip().lower()
        if g in ["male", "m"]:
            return "male"
        elif g in ["female", "f"]:
            return "female"
        return "unknown"

    df["gender"] = df["gender"].apply(map_gender)

    # Rule (c): Age filtering and city median imputation
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    valid_age = age_numeric.where((age_numeric >= 10) & (age_numeric <= 100), np.nan)
    city_median_age = valid_age.groupby(df["city"]).transform("median")
    df["age"] = valid_age.fillna(city_median_age).astype(int)

    # Rule (d): Datetime conversion
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    return df

def clean_orders_data(raw_orders_df, clean_products_df):
    """
    Cleans raw orders dataframe according to Part A Task 4 rules in exact order:
    1. Convert order_ts to datetime.
    2. Remove exact duplicate rows.
    3. Keep only the record with the latest order_ts per order_id.
    4. Convert order_amount to numeric, impute unparseables using product price, quantity, and discount.
    5. Set negative delivery_minutes to missing. For completed orders, impute missing with median
       delivery time of the order's product category among completed orders. For cancelled orders,
       leave delivery_minutes as missing (NaN).
    """
    df = raw_orders_df.copy()

    # Rule 1: Convert order_ts to datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # Rule 2: Remove exact duplicate rows
    df = df.drop_duplicates()

    # Rule 3: Keep latest order_ts per order_id
    df = df.sort_values("order_ts").groupby("order_id", as_index=False).last()

    # Rule 4: Convert order_amount and impute missing
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    price_map = clean_products_df.set_index("product_id")["price_inr"]
    imputed_amounts = (
        df["product_id"].map(price_map) * df["quantity"] * (1.0 - df["discount_pct"] / 100.0)
    ).round(2)
    df["order_amount"] = df["order_amount"].fillna(imputed_amounts)

    # Rule 5: Handle delivery_minutes
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df["delivery_minutes"] = df["delivery_minutes"].where(df["delivery_minutes"] >= 0, np.nan)

    cat_map = clean_products_df.set_index("product_id")["category"]
    df["_category"] = df["product_id"].map(cat_map)

    completed_mask = df["status"] == "completed"
    median_category_deliv = (
        df.loc[completed_mask]
        .groupby("_category")["delivery_minutes"]
        .transform("median")
    )
    df.loc[completed_mask, "delivery_minutes"] = (
        df.loc[completed_mask, "delivery_minutes"].fillna(median_category_deliv)
    )

    # Cleanup temporary category column
    df = df.drop(columns=["_category"])

    return df

def run_cleaning(raw_dir=os.path.join("data", "raw"), clean_dir=os.path.join("data", "clean")):
    os.makedirs(clean_dir, exist_ok=True)

    raw_cust = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    raw_prod = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    raw_ord = pd.read_csv(os.path.join(raw_dir, "orders.csv"))

    clean_cust = clean_customers_data(raw_cust)
    clean_prod = raw_prod.copy()
    clean_ord = clean_orders_data(raw_ord, clean_prod)

    # Write clean CSVs
    clean_cust.to_csv(os.path.join(clean_dir, "customers.csv"), index=False, date_format="%Y-%m-%d")
    clean_prod.to_csv(os.path.join(clean_dir, "products.csv"), index=False)
    clean_ord.to_csv(os.path.join(clean_dir, "orders.csv"), index=False, date_format="%Y-%m-%d %H:%M:%S")

    return clean_cust, clean_prod, clean_ord

if __name__ == "__main__":
    c_cust, c_prod, c_ord = run_cleaning()
    print("Cleaned tables saved to data/clean/")
    print(f"customers: {c_cust.shape}, products: {c_prod.shape}, orders: {c_ord.shape}")
