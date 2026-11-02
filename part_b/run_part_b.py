"""Run every Part B task and persist a JSON report of the measured results."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from part_b.rfm import compute_rfm_table
from part_b.clustering import evaluate_kmeans_clusters, build_segments
from part_b.lapse_modeling import build_cohort_and_features, run_lapse_experiment

CLUSTER_CHOICE = (
    "Choose K=4: the large inertia reduction from K=3 to K=4 is followed by smaller "
    "improvements, and K=4 has the strongest silhouette among the actionable choices "
    "K=3,4,5. Its four profiles support distinct occasional-buyer, premium, frequent-shopper "
    "and win-back campaigns. K-means inertia is the within-cluster squared-distance objective; "
    "adding centres increases flexibility, so the best achievable inertia cannot increase "
    "and the observed K=8 value is lower than K=2. Lower inertia alone cannot choose K because "
    "ever-smaller clusters may be less interpretable and less well separated. Cluster numbers "
    "are arbitrary labels and have no numerical ordering."
)


def write_part_b_report(rfm, evaluation, profile, pca_ratios, modeling, output_path=Path("outputs/part_b_report.json")):
    report = {
        "snapshot": "2025-10-01 00:00:00", "monetary_currency": "INR",
        "rfm_summary": {
            "customers": int(len(rfm)), "recency_median": float(rfm["recency"].median()),
            "recency_min": int(rfm["recency"].min()), "recency_max": int(rfm["recency"].max()),
            "frequency_mean": float(rfm["frequency"].mean()), "frequency_max": int(rfm["frequency"].max()),
            "monetary_mean_inr": float(rfm["monetary"].mean()), "monetary_max_inr": float(rfm["monetary"].max()),
        },
        "kmeans_evaluation": evaluation.to_dict(orient="records"), "chosen_k": 4,
        "cluster_choice_explanation": CLUSTER_CHOICE,
        "segment_profiles": profile.to_dict(orient="records"),
        "pca_explained_variance_ratio": pca_ratios.tolist(),
        "pca_total_explained_variance": float(sum(pca_ratios)),
        "lapse_experiment": modeling,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    # A script launched while inside part_b must still resolve data/ and outputs/.
    os.chdir(ROOT)
    rfm = compute_rfm_table()
    evaluation, _, _ = evaluate_kmeans_clusters(rfm)
    segments, profile, pca_ratios = build_segments(rfm, k=4)
    features = build_cohort_and_features()
    modeling = run_lapse_experiment(features)
    report = write_part_b_report(rfm, evaluation, profile, pca_ratios, modeling)
    print("PART B — Customer Segments and Lapse Prediction")
    print("RFM (monetary amounts in INR):", report["rfm_summary"])
    print("\nK-means evaluation:\n", evaluation.to_string(index=False))
    print(CLUSTER_CHOICE)
    print("\nSegment profiles:\n", profile.to_string(index=False))
    print("PCA explained-variance ratios:", pca_ratios, "total:", sum(pca_ratios))
    print(f"\nCohort: {modeling['cohort_size']}; lapsed: {modeling['lapsed_count']} ({modeling['lapsed_pct']:.2f}%); train/test: {modeling['train_size']}/{modeling['test_size']}")
    print("Frequency mean from fitted LR scaler:", modeling["learned_freq_mean"], "; full cohort:", modeling["full_cohort_freq_mean"])
    print(modeling["scaler_explanation"])
    print("Missing feature values:", not modeling["no_missing_feature_values"])
    print("\nBaseline (lapsed=1):", modeling["baseline"])
    for key, label in [("lr", "Logistic regression"), ("rf", "Random forest")]:
        print(f"{label} grid/CV:", modeling[f"{key}_tuning"])
        print(f"{label} test (lapsed=1):", modeling[f"{key}_test"])
    print(modeling["selected_model_justification"])
    print("\nBalanced training class counts:", modeling["balanced_train_class_counts"])
    print("Extra lapsed duplicates:", modeling["oversampled_duplicate_count"])
    print("Balanced LR test:", modeling["lr_balanced_test"])
    print("Metric changes:", modeling["balance_metric_changes"])
    print(modeling["balance_explanation"])
    print("\nRF five most important features:\n", pd.DataFrame(modeling["top5_features"]).to_string(index=False))
    print("LR coefficient signs:\n", pd.DataFrame(modeling["lr_coefficient_readings"]).to_string(index=False))
    print("Reload reproduces test predictions exactly:", modeling["reload_predictions_match"])
    print("\nPremium test retention list:\n", pd.DataFrame(modeling["retention_top10"]).to_string(index=False))
    print("Actually lapsed:", modeling["retention_actually_lapsed_count"], "/", len(modeling["retention_top10"]))
    print(modeling["retention_limitation"])
    print("\nWrote outputs/customer_segments.csv, outputs/lapse_model.joblib, outputs/retention_list.csv and outputs/part_b_report.json")
    return report


if __name__ == "__main__":
    main()
