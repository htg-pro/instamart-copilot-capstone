"""part_b/lapse_modeling.py - Feature engineering, pipeline training, tuning, and evaluation for customer lapse prediction."""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def build_cohort_and_features(orders_path=os.path.join("data", "clean", "orders.csv"),
                              customers_path=os.path.join("data", "clean", "customers.csv"),
                              products_path=os.path.join("data", "clean", "products.csv"),
                              cutoff_date="2025-09-01 00:00:00"):
    orders = pd.read_csv(orders_path)
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    cutoff = pd.Timestamp(cutoff_date)

    # Attach category from products
    cat_map = products.set_index("product_id")["category"]
    orders["category"] = orders["product_id"].map(cat_map)

    # Strictly filter orders before cutoff
    pre_orders = orders[orders["order_ts"] < cutoff].copy()
    pre_completed = pre_orders[pre_orders["status"] == "completed"].copy()

    # Cohort: customers with at least one completed order before cutoff
    cohort_cids = sorted(pre_completed["customer_id"].unique())

    # Ground truth label: 1 if NO completed order in September 2025 [2025-09-01 through 2025-09-30]
    post_completed = orders[(orders["order_ts"] >= cutoff) & (orders["status"] == "completed")].copy()
    active_in_sept_cids = set(post_completed["customer_id"].unique())
    y_series = pd.Series([1 if cid not in active_in_sept_cids else 0 for cid in cohort_cids], index=cohort_cids)

    # Pre-cutoff features
    rec = pre_completed.groupby("customer_id")["order_ts"].max().apply(lambda x: int((cutoff - x).days))
    freq = pre_completed.groupby("customer_id")["order_id"].nunique()
    mon = pre_completed.groupby("customer_id")["order_amount"].sum()
    avg_disc = pre_completed.groupby("customer_id")["discount_pct"].mean()

    # Cancellation rate before cutoff: cancelled orders / all orders with final status
    total_pre = pre_orders.groupby("customer_id")["order_id"].nunique()
    canc_pre = pre_orders[pre_orders["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    canc_rate = (canc_pre / total_pre).fillna(0.0)

    avg_deliv = pre_completed.groupby("customer_id")["delivery_minutes"].mean()
    fav_cat = pre_completed.groupby("customer_id")["category"].agg(lambda x: x.mode()[0])

    features_df = pd.DataFrame({
        "customer_id": cohort_cids,
        "recency": [rec[c] for c in cohort_cids],
        "frequency": [freq[c] for c in cohort_cids],
        "monetary": [round(mon[c], 2) for c in cohort_cids],
        "avg_discount_pct": [round(avg_disc[c], 2) for c in cohort_cids],
        "cancellation_rate": [round(canc_rate[c], 4) for c in cohort_cids],
        "avg_delivery_minutes": [round(avg_deliv[c], 2) for c in cohort_cids],
        "favourite_category": [fav_cat[c] for c in cohort_cids]
    }).merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")

    features_df["tenure_days"] = (cutoff - features_df["signup_date"]).dt.days
    features_df["lapsed"] = [y_series[c] for c in features_df["customer_id"]]

    return features_df

def run_lapse_experiment(features_df, output_model_path=os.path.join("outputs", "lapse_model.joblib"),
                         segments_path=os.path.join("outputs", "customer_segments.csv"),
                         retention_path=os.path.join("outputs", "retention_list.csv")):
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(retention_path), exist_ok=True)

    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
    ]
    categorical_cols = ["favourite_category", "city", "gender"]

    # Check for missing values in features
    assert features_df[numeric_cols + categorical_cols].isna().sum().sum() == 0, "Missing values detected in features!"

    X = features_df[numeric_cols + categorical_cols]
    y = features_df["lapsed"]

    # Train / test split (stratified, 80/20, random_state=42)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features_df.index, test_size=0.2, stratify=y, random_state=42
    )

    # Check frequency scaler difference
    train_scaler = StandardScaler().fit(X_train[["frequency"]])
    learned_freq_mean = train_scaler.mean_[0]
    full_cohort_freq_mean = X["frequency"].mean()

    # Preprocessor: fitted on training set only
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
    ])

    # 1. Baseline: Predict majority class (0 - not lapsed)
    y_pred_baseline = np.zeros(len(y_test))
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)

    # 2. Logistic Regression Pipeline & Grid Search
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)

    y_pred_lr = grid_lr.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)

    # 3. Random Forest Pipeline & Grid Search
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)

    y_pred_rf = grid_rf.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

    # 4. Class-balance experiment on Logistic Regression
    train_df = pd.concat([X_train, y_train], axis=1)
    non_lapsed = train_df[train_df["lapsed"] == 0]
    lapsed = train_df[train_df["lapsed"] == 1]

    # Oversample minority class (lapsed) with replacement to match non_lapsed count
    lapsed_oversampled = lapsed.sample(n=len(non_lapsed), replace=True, random_state=42)
    train_balanced = pd.concat([non_lapsed, lapsed_oversampled], axis=0)

    X_train_bal = train_balanced.drop(columns=["lapsed"])
    y_train_bal = train_balanced["lapsed"]

    best_lr_C = grid_lr.best_params_["clf__C"]
    pipe_lr_bal = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(C=best_lr_C, max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)

    cm_lr_bal = confusion_matrix(y_test, y_pred_lr_bal)
    prec_lr_bal = precision_score(y_test, y_pred_lr_bal, zero_division=0)
    rec_lr_bal = recall_score(y_test, y_pred_lr_bal, zero_division=0)
    f1_lr_bal = f1_score(y_test, y_pred_lr_bal, zero_division=0)

    # 5. Feature Importance & Interpretation
    best_rf_model = grid_rf.best_estimator_
    fitted_preprocessor = best_rf_model.named_steps["pre"]
    cat_feature_names = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    rf_importances = best_rf_model.named_steps["clf"].feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": rf_importances
    }).sort_values("importance", ascending=False)
    top5_features = feat_imp_df.head(5)

    # Logistic regression coefficients
    best_lr_clf = grid_lr.best_estimator_.named_steps["clf"]
    lr_coefs = dict(zip(all_feature_names, best_lr_clf.coef_[0]))

    # 6. Save best model and verify reload
    selected_model = grid_rf.best_estimator_
    joblib.dump(selected_model, output_model_path)
    reloaded_model = joblib.load(output_model_path)
    reloaded_pred = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_pred), "Reloaded model predictions mismatch!"

    # 7. Retention List for Premium customers in test set
    test_customers = features_df.loc[idx_test].copy()
    test_customers["lapse_probability"] = selected_model.predict_proba(X_test)[:, 1].round(4)
    test_customers["actually_lapsed"] = y_test.values

    # Merge segment info
    if os.path.exists(segments_path):
        seg_df = pd.read_csv(segments_path)
        test_customers = test_customers.merge(seg_df[["customer_id", "is_premium", "segment_name"]], on="customer_id", how="left")
    else:
        test_customers["is_premium"] = False

    premium_test = test_customers[test_customers["is_premium"] == True].copy()
    retention_top10 = premium_test.sort_values("lapse_probability", ascending=False).head(10)
    retention_output = retention_top10[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_output.to_csv(retention_path, index=False)

    return {
        "cohort_size": len(features_df),
        "lapsed_count": int(y.sum()),
        "lapsed_pct": round(float(y.mean() * 100), 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "learned_freq_mean": round(learned_freq_mean, 4),
        "full_cohort_freq_mean": round(full_cohort_freq_mean, 4),
        "baseline": {"accuracy": round(baseline_acc, 4), "f1": round(baseline_f1, 4)},
        "lr_tuning": {"best_params": grid_lr.best_params_, "best_cv_f1": round(grid_lr.best_score_, 4)},
        "lr_test": {"cm": cm_lr, "precision": round(prec_lr, 4), "recall": round(rec_lr, 4), "f1": round(f1_lr, 4)},
        "rf_tuning": {"best_params": grid_rf.best_params_, "best_cv_f1": round(grid_rf.best_score_, 4)},
        "rf_test": {"cm": cm_rf, "precision": round(prec_rf, 4), "recall": round(rec_rf, 4), "f1": round(f1_rf, 4)},
        "lr_balanced_test": {"cm": cm_lr_bal, "precision": round(prec_lr_bal, 4), "recall": round(rec_lr_bal, 4), "f1": round(f1_lr_bal, 4)},
        "top5_features": top5_features,
        "lr_coefs": lr_coefs,
        "retention_top10": retention_output,
        "retention_actually_lapsed_count": int(retention_output["actually_lapsed"].sum())
    }

if __name__ == "__main__":
    feat_df = build_cohort_and_features()
    res = run_lapse_experiment(feat_df)
    print("=== Part B Lapse Experiment Summary ===")
    print(f"Cohort size: {res['cohort_size']}, Lapsed count: {res['lapsed_count']} ({res['lapsed_pct']}%)")
    print(f"Train/Test sizes: {res['train_size']} / {res['test_size']}")
    print(f"Learned train freq mean: {res['learned_freq_mean']} vs Full cohort: {res['full_cohort_freq_mean']}")
    print(f"LR Best params: {res['lr_tuning']['best_params']}, CV F1: {res['lr_tuning']['best_cv_f1']}")
    print(f"LR Test - Prec: {res['lr_test']['precision']}, Rec: {res['lr_test']['recall']}, F1: {res['lr_test']['f1']}")
    print(f"RF Best params: {res['rf_tuning']['best_params']}, CV F1: {res['rf_tuning']['best_cv_f1']}")
    print(f"RF Test - Prec: {res['rf_test']['precision']}, Rec: {res['rf_test']['recall']}, F1: {res['rf_test']['f1']}")
    print(f"LR Balanced Test - Prec: {res['lr_balanced_test']['precision']}, Rec: {res['lr_balanced_test']['recall']}, F1: {res['lr_balanced_test']['f1']}")
    print("\nTop 5 Features (Random Forest):")
    print(res["top5_features"])
    print(f"\nRetention Top 10 (Premium test customers):\n{res['retention_top10']}")
    print(f"Actually lapsed count in retention list: {res['retention_actually_lapsed_count']}")
