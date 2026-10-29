"""part_a/db.py - SQLite storage and SQL query execution for Part A."""
import os
import sqlite3
import pandas as pd

def store_to_sqlite(clean_cust_df, clean_prod_df, clean_ord_df, db_path=os.path.join("data", "instamart.db")):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Format dates as strings
    cust_to_store = clean_cust_df.copy()
    if pd.api.types.is_datetime64_any_dtype(cust_to_store["signup_date"]):
        cust_to_store["signup_date"] = cust_to_store["signup_date"].dt.strftime("%Y-%m-%d")

    ord_to_store = clean_ord_df.copy()
    if pd.api.types.is_datetime64_any_dtype(ord_to_store["order_ts"]):
        ord_to_store["order_ts"] = ord_to_store["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cust_to_store.to_sql("customers", conn, if_exists="replace", index=False)
    clean_prod_df.to_sql("products", conn, if_exists="replace", index=False)
    ord_to_store.to_sql("orders", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    return db_path

def run_sql_queries(db_path=os.path.join("data", "instamart.db"), sql_path=os.path.join("sql", "queries.sql")):
    conn = sqlite3.connect(db_path)

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Remove all single-line SQL comments before splitting
    no_comments = "\n".join(
        line for line in sql_content.splitlines() if not line.strip().startswith("--")
    )

    # Split by semicolon
    raw_statements = [stmt.strip() for stmt in no_comments.split(";") if stmt.strip()]

    results = {}
    for idx, stmt in enumerate(raw_statements, start=1):
        df = pd.read_sql_query(stmt, conn)
        results[f"Q{idx}"] = df

    conn.close()
    return results

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from part_a.clean import run_cleaning
    c_cust, c_prod, c_ord = run_cleaning()
    db = store_to_sqlite(c_cust, c_prod, c_ord)
    print(f"Data stored in {db}")
    res = run_sql_queries()
    for q, df in res.items():
        print(f"{q}: {len(df)} rows returned")
