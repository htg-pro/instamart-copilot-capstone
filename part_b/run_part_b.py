"""part_b/run_part_b.py - End-to-end execution of Part B."""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_b.rfm import compute_rfm_table
from part_b.clustering import evaluate_kmeans_clusters, build_segments
from part_b.lapse_modeling import build_cohort_and_features, run_lapse_experiment

def main():
    print("==================================================")
    print("   PART B: CUSTOMER SEGMENTS & LAPSE PREDICTION   ")
    print("==================================================")

    # 1. RFM Computation
    print("\n--- Step 1: Computing RFM Table ---")
    rfm = compute_rfm_table()
    print(f"Total customers evaluated: {len(rfm)}")
    print(f"Recency: median={rfm['recency'].median()}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")

    # 2. KMeans Clustering
    print("\n--- Step 2: Evaluating KMeans Clusters (K=2 to 8) ---")
    eval_df, _, _ = evaluate_kmeans_clusters(rfm)
    print(eval_df.to_string(index=False))

    print("\n--- Step 3: Segmenting Customers (K=4) & PCA Projection ---")
    segments_df, cluster_stats, pca_ratios = build_segments(rfm, k=4)
    print("\nCluster Statistics:")
    print(cluster_stats.to_string(index=False))
    print(f"\nPCA Explained Variance Ratio: Component 1 = {pca_ratios[0]:.4f}, Component 2 = {pca_ratios[1]:.4f}, Total = {sum(pca_ratios):.4f}")
    print("Wrote outputs/customer_segments.csv")

    # 3. Lapse Feature Engineering & Modeling
    print("\n--- Step 4: Building Lapse Cohort & Features (Cutoff: 2025-09-01) ---")
    features_df = build_cohort_and_features()
    print(f"Cohort Size: {len(features_df)}")
    print(f"Lapsed Customers: {features_df['lapsed'].sum()} ({features_df['lapsed'].mean()*100:.2f}%)")

    print("\n--- Step 5: Training, Tuning & Evaluating Models ---")
    results = run_lapse_experiment(features_df)
    print(f"Train rows: {results['train_size']}, Test rows: {results['test_size']}")
    print(f"Learned train frequency mean: {results['learned_freq_mean']} vs Full cohort: {results['full_cohort_freq_mean']}")
    print(f"Majority Baseline Test Accuracy: {results['baseline']['accuracy']*100:.2f}%, Lapsed F1: {results['baseline']['f1']}")

    print(f"\nLogistic Regression Best Params: {results['lr_tuning']['best_params']}, Best CV F1: {results['lr_tuning']['best_cv_f1']:.4f}")
    print(f"Logistic Regression Test - Precision: {results['lr_test']['precision']:.4f}, Recall: {results['lr_test']['recall']:.4f}, F1: {results['lr_test']['f1']:.4f}")

    print(f"\nRandom Forest Best Params: {results['rf_tuning']['best_params']}, Best CV F1: {results['rf_tuning']['best_cv_f1']:.4f}")
    print(f"Random Forest Test - Precision: {results['rf_test']['precision']:.4f}, Recall: {results['rf_test']['recall']:.4f}, F1: {results['rf_test']['f1']:.4f}")

    print(f"\nClass-Balanced Logistic Regression Test - Precision: {results['lr_balanced_test']['precision']:.4f}, Recall: {results['lr_balanced_test']['recall']:.4f}, F1: {results['lr_balanced_test']['f1']:.4f}")

    print("\n--- Step 6: Feature Importances & Model Interpretability ---")
    print("Top 5 Features (Random Forest):")
    print(results["top5_features"].to_string(index=False))

    print("\nWrote outputs/lapse_model.joblib")
    print("Wrote outputs/retention_list.csv (Top 10 at-risk premium customers):")
    print(results["retention_top10"].to_string(index=False))
    print(f"Actually lapsed count in retention list: {results['retention_actually_lapsed_count']} / 10")

    print("\nPart B completed successfully!")

if __name__ == "__main__":
    main()
