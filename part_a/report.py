"""Measured Part A results and checks shared by the CLI and notebook."""
import json
from pathlib import Path

import pandas as pd

from part_a.audit import audit_raw_data
from part_a.db import run_sql_queries
from part_a.eda import run_eda


def cleaning_changes(raw_dir="data/raw"):
    """Count changes at the exact stage where each cleaning rule applies."""
    customers = pd.read_csv(Path(raw_dir) / "customers.csv")
    orders = pd.read_csv(Path(raw_dir) / "orders.csv")
    products = pd.read_csv(Path(raw_dir) / "products.csv")
    city = customers.city.str.strip().str.title()
    age = pd.to_numeric(customers.age, errors="coerce")
    invalid_age = age.notna() & ~age.between(10, 100)
    gender = customers.gender.fillna("").str.strip().str.lower().map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    ).fillna("unknown")
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    unique = orders.drop_duplicates()
    latest = (unique.sort_values("order_ts", kind="stable")
              .drop_duplicates("order_id", keep="last").copy())
    minutes = pd.to_numeric(latest.delivery_minutes, errors="coerce")
    completed = latest.status.eq("completed")
    missing_minutes = minutes.mask(minutes.lt(0)).isna()
    return {
        "city_values_standardized": int(customers.city.ne(city).sum()),
        "gender_values_standardized": int(customers.gender.fillna("").ne(gender).sum()),
        "blank_gender_to_unknown": int(customers.gender.isna().sum()),
        "invalid_ages_to_missing": int(invalid_age.sum()),
        "ages_imputed_city_median": int((age.isna() | invalid_age).sum()),
        "signup_dates_parsed": len(customers),
        "order_timestamps_parsed": len(orders),
        "exact_duplicate_rows_removed": len(orders) - len(unique),
        "older_order_records_removed": len(unique) - len(latest),
        "amounts_reconstructed": int(pd.to_numeric(latest.order_amount, errors="coerce").isna().sum()),
        "negative_delivery_to_missing": int(minutes.lt(0).sum()),
        "completed_delivery_imputed": int((completed & missing_minutes).sum()),
        "cancelled_delivery_left_missing": int(latest.status.eq("cancelled").sum()),
        "products_changed": 0,
        "category_delivery_medians": {
            str(k): float(v) for k, v in minutes.mask(minutes.lt(0))[completed]
            .groupby(latest.product_id.map(products.set_index("product_id").category)[completed])
            .median().items()
        },
        "city_age_medians": {str(k): float(v) for k, v in age.mask(invalid_age).groupby(city).median().items()},
    }


def validate_sql_results(results):
    """Check every SQL business checkpoint, including tie behavior."""
    assert len(results) == 7
    assert len(results["Q1"]) == 30
    q2 = results["Q2"]
    assert len(q2) == 30 and abs(q2.revenue.sum() - 2998668.80) < .01
    maximum = q2.loc[q2.revenue.idxmax()]
    assert (maximum.city, maximum.category) == ("Bengaluru", "Electronics")
    assert abs(maximum.revenue - 233909.45) < .01
    assert len(results["Q3"]) == 11 and len(results["Q4"]) == 25
    assert len(results["Q5"]) == 550
    q6 = results["Q6"]
    expected = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40),
                "Delhi": ("C0291", 13742.60), "Hyderabad": ("C0743", 12730.05),
                "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
    for row in q6[q6.row_num.eq(1)].itertuples():
        customer_id, spend = expected[row.city]
        assert row.customer_id == customer_id and abs(row.total_spend - spend) < .01
    assert int(q6.rank_num.ne(q6.row_num).sum()) == 8
    q7 = results["Q7"]
    assert len(q7) == 183 and abs(q7.cumulative_revenue.iloc[-1] - 2998668.80) < .01
    return True


def build_report(output_path="outputs/part_a_report.json"):
    summary, metrics, columns = audit_raw_data()
    queries = run_sql_queries()
    validate_sql_results(queries)
    eda = run_eda()
    assert abs(eda["grand_total"] - queries["Q2"].revenue.sum()) < .01
    customers = pd.read_csv("data/clean/customers.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    completed = orders[orders.status.eq("completed")]
    correlation = eda["correlation_matrix"]
    pairs = [(a, b, float(correlation.loc[a, b]))
             for i, a in enumerate(correlation.columns) for b in correlation.columns[i + 1:]]
    pair = max(pairs, key=lambda item: abs(item[2]))
    report = {
        "raw_shapes": summary, "raw_audit": metrics,
        "column_audit": {k: v.to_dict("records") for k, v in columns.items()},
        "cleaning_changes": cleaning_changes(),
        "clean": {"customers": len(customers), "products": 40, "orders": len(orders),
                  "city_counts": customers.city.value_counts().to_dict(),
                  "gender_counts": customers.gender.value_counts().to_dict(),
                  "age_min": float(customers.age.min()), "age_max": float(customers.age.max()),
                  "completed": len(completed), "cancelled": int(orders.status.eq("cancelled").sum()),
                  "completed_revenue_inr": round(float(completed.order_amount.sum()), 2)},
        "sql": {k: {"rows": len(v), "results": v.to_dict("records")} for k, v in queries.items()},
        "average_customer_spend_inr": round(float(completed.groupby("customer_id").order_amount.sum().mean()), 2),
        "pivot_table": eda["pivot_table"].reset_index().to_dict("records"),
        "city_aggregation": eda["city_aggregation"].reset_index().to_dict("records"),
        "distribution": eda["distribution_stats"],
        "correlation_matrix": correlation.to_dict(),
        "largest_absolute_correlation": {"features": list(pair[:2]), "correlation": pair[2]},
        "checkpoints_passed": True,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, frame in queries.items():
        frame.to_csv(path.parent / f"sql_{name.lower()}.csv", index=False)
    return report
