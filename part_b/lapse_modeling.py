"""Leakage-free pre-September features, train-only tuning and lapse evaluation.

September orders define the target only. Whole-window Premium membership is
used solely for the illustrative retention export, never as a model feature.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = [
    "recency", "frequency", "monetary", "avg_discount_pct",
    "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days",
]
CATEGORICAL_COLUMNS = ["favourite_category", "city", "gender"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
LR_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100]}
RF_GRID = {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}
RETENTION_LIMITATION = (
    "The Premium flag is a post-hoc whole-window RFM label that includes September; "
    "therefore the retention list is illustrative and is not a leakage-free evaluation."
)


def build_cohort_and_features(
    orders_path=Path("data/clean/orders.csv"),
    customers_path=Path("data/clean/customers.csv"),
    products_path=Path("data/clean/products.csv"),
    cutoff_date="2025-09-01 00:00:00",
):
    """Build one row per customer active before cutoff; label the next month."""
    orders = pd.read_csv(orders_path)
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    cutoff = pd.Timestamp(cutoff_date)
    label_end = cutoff + pd.DateOffset(months=1)
    orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])

    pre_orders = orders.loc[orders["order_ts"] < cutoff].copy()
    assert (pre_orders["order_ts"] < cutoff).all(), "Post-cutoff orders entered features."
    completed = pre_orders.loc[pre_orders["status"] == "completed"]
    group = completed.groupby("customer_id", sort=True)
    features = group.agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=("category", lambda values: values.mode().iloc[0]),
    )
    features["recency"] = (cutoff - features.pop("last_order_ts")).dt.days
    # Only final statuses belong in this denominator. Pending orders do not.
    final = pre_orders.loc[pre_orders["status"].isin(["completed", "cancelled"])]
    totals = final.groupby("customer_id")["order_id"].nunique()
    cancelled = final.loc[final["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    features["cancellation_rate"] = cancelled.reindex(totals.index, fill_value=0).div(totals).reindex(features.index)
    features = features.reset_index().merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]],
        on="customer_id", how="left", validate="one_to_one", sort=False,
    )
    features["tenure_days"] = (cutoff - features["signup_date"]).dt.days

    # Use a closed September window: later months cannot change this label.
    label_orders = orders.loc[
        orders["order_ts"].ge(cutoff) & orders["order_ts"].lt(label_end)
        & orders["status"].eq("completed")
    ]
    features["lapsed"] = (~features["customer_id"].isin(label_orders["customer_id"])).astype(int)
    assert not features[FEATURE_COLUMNS].isna().any().any(), "Missing feature values."
    assert not {"segment_id", "segment_name", "is_premium"}.intersection(FEATURE_COLUMNS)
    features.attrs["feature_cutoff"] = str(cutoff)
    features.attrs["maximum_feature_order_ts"] = str(pre_orders["order_ts"].max())
    features.attrs["label_window_end_exclusive"] = str(label_end)
    return features


def split_lapse_data(features_df):
    """One reproducible split shared by the notebook and command-line runner."""
    assert not features_df[FEATURE_COLUMNS].isna().any().any()
    X = features_df[FEATURE_COLUMNS]
    y = features_df["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    assert set(X_train.index).isdisjoint(X_test.index)
    return X_train, X_test, y_train, y_test


def make_preprocessor():
    """Return fresh transformers: each pipeline/CV fold learns training rows only."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])


def positive_class_metrics(y_true, predictions):
    """All precision/recall/F1 refer explicitly to lapsed=1."""
    return {
        "cm": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
    }


def balance_training_only(X_train, y_train):
    """Preserve EVERY original row, then add only the missing minority duplicates."""
    counts = y_train.value_counts()
    assert counts[0] >= counts[1], "This experiment assumes lapsed=1 is the minority."
    extra_count = int(counts[0] - counts[1])
    extra_indices = y_train.loc[y_train.eq(1)].sample(n=extra_count, replace=True, random_state=42).index
    X_balanced = pd.concat([X_train, X_train.loc[extra_indices]])
    y_balanced = pd.concat([y_train, y_train.loc[extra_indices]])
    pd.testing.assert_frame_equal(X_balanced.iloc[:len(X_train)], X_train)
    pd.testing.assert_series_equal(y_balanced.iloc[:len(y_train)], y_train)
    assert y_balanced.value_counts().nunique() == 1
    return X_balanced, y_balanced


def feature_reading(feature, coefficient=None):
    """Short conditional association readings; these are not causal claims."""
    description = {
        "recency": "Days since the last completed order indicate disengagement.",
        "frequency": "Earlier completed-order frequency distinguishes regular shoppers.",
        "monetary": "Earlier spending captures customer value and purchase intensity in INR.",
        "avg_discount_pct": "Earlier discount usage helps identify promotion-sensitive shoppers.",
        "avg_delivery_minutes": "Earlier delivery duration captures service experience.",
        "cancellation_rate": "Earlier cancelled-order share captures purchase friction.",
        "age": "Customer age can be associated with different shopping patterns.",
        "tenure_days": "Time since signup indicates how established the relationship is.",
    }.get(feature, "This encoded customer attribute helps distinguish shopping patterns.")
    if coefficient is not None:
        direction = "higher" if coefficient > 0 else "lower" if coefficient < 0 else "unchanged"
        description += f" Its coefficient implies {direction} lapse log-odds as the feature increases, holding other inputs fixed."
    return description


def run_lapse_experiment(
    features_df,
    output_model_path=Path("outputs/lapse_model.joblib"),
    segments_path=Path("outputs/customer_segments.csv"),
    retention_path=Path("outputs/retention_list.csv"),
):
    """Tune on training rows/CV, evaluate once, balance train only and persist."""
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(retention_path).parent.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_lapse_data(features_df)
    searches = {}
    for name, classifier, parameter_grid in [
        ("lr", LogisticRegression(max_iter=1000, random_state=42), LR_GRID),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42), RF_GRID),
    ]:
        pipeline = Pipeline([("pre", make_preprocessor()), ("clf", classifier)])
        search = GridSearchCV(pipeline, parameter_grid, cv=5, scoring="f1")
        search.fit(X_train, y_train)
        searches[name] = search

    # Choose the tuned base model by training cross-validation before test scoring.
    selected_key = max(searches, key=lambda name: searches[name].best_score_)
    selected_model = searches[selected_key].best_estimator_
    predictions = {name: search.predict(X_test) for name, search in searches.items()}
    metrics = {name: positive_class_metrics(y_test, pred) for name, pred in predictions.items()}
    assert metrics[selected_key]["f1"] >= 0.55, "Selected model misses required positive-class F1."

    X_balanced, y_balanced = balance_training_only(X_train, y_train)
    assert set(X_balanced.index).isdisjoint(X_test.index), "A test row was resampled."
    # Cloning removes fitted state; the balanced experiment cannot alter either base model.
    balanced_lr = clone(searches["lr"].best_estimator_).fit(X_balanced, y_balanced)
    balanced_metrics = positive_class_metrics(y_test, balanced_lr.predict(X_test))
    metric_changes = {name: balanced_metrics[name] - metrics["lr"][name] for name in ["precision", "recall", "f1"]}
    most_changed = max(metric_changes, key=lambda name: abs(metric_changes[name]))

    best_rf = searches["rf"].best_estimator_
    rf_names = best_rf.named_steps["pre"].get_feature_names_out().tolist()
    importance_df = pd.DataFrame({
        "feature": [name.split("__", 1)[-1] for name in rf_names],
        "importance": best_rf.named_steps["clf"].feature_importances_,
    }).sort_values("importance", ascending=False).head(5)
    best_lr = searches["lr"].best_estimator_
    lr_names = best_lr.named_steps["pre"].get_feature_names_out().tolist()
    coefficients = dict(zip([name.split("__", 1)[-1] for name in lr_names], best_lr.named_steps["clf"].coef_[0].tolist()))
    coefficient_readings = [{
        "feature": name, "coefficient": coefficients[name],
        "sign": "positive" if coefficients[name] > 0 else "negative" if coefficients[name] < 0 else "zero",
        "business_reading": feature_reading(name, coefficients[name]),
    } for name in ["recency", "frequency", "avg_discount_pct"]]

    joblib.dump(selected_model, output_model_path)
    reloaded_model = joblib.load(output_model_path)
    reload_matches = np.array_equal(predictions[selected_key], reloaded_model.predict(X_test))
    assert reload_matches, "Reloaded model predictions differ."
    classes = selected_model.named_steps["clf"].classes_.tolist()
    positive_column = classes.index(1)
    test_customers = features_df.loc[X_test.index, ["customer_id"]].copy()
    test_customers["lapse_probability"] = selected_model.predict_proba(X_test)[:, positive_column]
    test_customers["actually_lapsed"] = y_test.to_numpy()
    segments = pd.read_csv(segments_path)
    test_customers = test_customers.merge(
        segments[["customer_id", "is_premium"]], on="customer_id", how="left", validate="one_to_one",
    )
    premium = test_customers["is_premium"].fillna(False).eq(True)
    retention = test_customers.loc[premium].sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True], kind="stable",
    ).head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention.to_csv(retention_path, index=False)

    selected_label = "random forest" if selected_key == "rf" else "logistic regression"
    selected_metrics = metrics[selected_key]
    tn, fp = selected_metrics["cm"][0]
    fn, tp = selected_metrics["cm"][1]
    justification = (
        f"We select the tuned {selected_label} because its training cross-validated F1 "
        f"is highest ({searches[selected_key].best_score_:.4f}); the test set did not choose its parameters. "
        f"On the untouched test set its lapsed-class F1 is {selected_metrics['f1']:.4f}, "
        f"precision is {selected_metrics['precision']:.4f} and recall is {selected_metrics['recall']:.4f}. "
        f"It misses {fn} lapsing customers and wrongly targets {fp} loyal customers at the default threshold. "
        "Missing a lapsing customer can lose repeat revenue, whereas an unnecessary retention offer costs "
        "discount spend and may annoy a loyal customer, so the team should validate campaign economics "
        "and a recall-oriented threshold on a separate validation set before deployment."
    )
    fitted_numeric = best_lr.named_steps["pre"].named_transformers_["num"]
    train_freq_mean = float(fitted_numeric.mean_[NUMERIC_COLUMNS.index("frequency")])
    result = {
        "cohort_size": int(len(features_df)), "lapsed_count": int(features_df["lapsed"].sum()),
        "lapsed_pct": float(features_df["lapsed"].mean() * 100),
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "learned_freq_mean": train_freq_mean, "full_cohort_freq_mean": float(features_df["frequency"].mean()),
        "scaler_explanation": "The scaler mean uses training customers only; the full-cohort mean also includes held-out customers, so random sampling makes them differ.",
        "feature_columns": FEATURE_COLUMNS, "no_missing_feature_values": True,
        "feature_audit": dict(features_df.attrs),
        "nominal_encoding": "OneHotEncoder(drop='first', handle_unknown='ignore') fitted inside each training pipeline/CV fold.",
        "baseline": positive_class_metrics(y_test, np.zeros(len(y_test), dtype=int)),
        "lr_tuning": {"grid": LR_GRID, "best_params": searches["lr"].best_params_, "best_cv_f1": float(searches["lr"].best_score_), "cv_fits": 25},
        "rf_tuning": {"grid": RF_GRID, "best_params": searches["rf"].best_params_, "best_cv_f1": float(searches["rf"].best_score_), "cv_fits": 45},
        "lr_test": metrics["lr"], "rf_test": metrics["rf"], "lr_balanced_test": balanced_metrics,
        "balanced_train_class_counts": {str(k): int(v) for k, v in y_balanced.value_counts().sort_index().items()},
        "oversampled_duplicate_count": int(len(X_balanced) - len(X_train)),
        "balance_metric_changes": metric_changes, "balance_metric_moved_most": most_changed,
        "balance_explanation": (
            f"{most_changed.capitalize()} moves most ({metric_changes[most_changed]:+.4f}); "
            "balancing increases the incentive to identify lapsing customers, but extra targeting can lower precision."
        ),
        "top5_features": [{**row, "business_reading": feature_reading(row["feature"])} for row in importance_df.to_dict(orient="records")],
        "lr_coefs": coefficients, "lr_coefficient_readings": coefficient_readings,
        "selected_model": selected_label, "selected_model_justification": justification,
        "reload_predictions_match": bool(reload_matches), "premium_test_customers": int(premium.sum()),
        "retention_top10": retention.to_dict(orient="records"),
        "retention_actually_lapsed_count": int(retention["actually_lapsed"].sum()),
        "retention_limitation": RETENTION_LIMITATION,
    }
    return result
