"""part_b/clustering.py - K-means clustering and segmentation on RFM metrics."""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def evaluate_kmeans_clusters(rfm_df, k_range=range(2, 9), random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(sil, 3)
        })

    eval_df = pd.DataFrame(results)
    return eval_df, scaler, X_scaled

def build_segments(rfm_df, k=4, output_path=os.path.join("outputs", "customer_segments.csv"), random_state=42):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    cluster_labels = km.fit_predict(X_scaled)

    segments_df = rfm_df.copy()
    segments_df["segment_id"] = cluster_labels

    # Identify cluster statistics to assign business names and premium flag
    cluster_stats = segments_df.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # The cluster with the highest mean monetary is marked as Premium
    premium_cluster_id = cluster_stats.loc[cluster_stats["mean_monetary"].idxmax(), "segment_id"]

    # Assign segment names
    # E.g.
    # cluster with highest monetary -> "High-Value Premium"
    # cluster with highest frequency / lowest recency -> "Active Frequent Shoppers"
    # cluster with highest recency -> "Dormant / At-Risk"
    # remaining cluster -> "Occasional / Regular Buyers"
    name_map = {}
    for _, row in cluster_stats.iterrows():
        cid = int(row["segment_id"])
        if cid == premium_cluster_id:
            name_map[cid] = "High-Value Premium"
        elif row["mean_frequency"] > 12.0:
            name_map[cid] = "Active Frequent Shoppers"
        elif row["mean_recency"] > 60.0:
            name_map[cid] = "Dormant / At-Risk"
        else:
            name_map[cid] = "Occasional / Regular Buyers"

    segments_df["segment_name"] = segments_df["segment_id"].map(name_map)
    segments_df["is_premium"] = segments_df["segment_id"] == premium_cluster_id

    # PCA projection
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    pca_ratios = pca.explained_variance_ratio_

    # Save to outputs/customer_segments.csv
    cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    segments_df[cols].to_csv(output_path, index=False)

    return segments_df, cluster_stats, pca_ratios

if __name__ == "__main__":
    from part_b.rfm import compute_rfm_table
    rfm = compute_rfm_table()
    eval_df, _, _ = evaluate_kmeans_clusters(rfm)
    print("=== KMeans Evaluation (K=2 to 8) ===")
    print(eval_df)
    segments_df, stats, pca_ratios = build_segments(rfm)
    print("\n=== Cluster Profile (K=4) ===")
    print(stats)
    print(f"\nPCA Explained Variance Ratio: {pca_ratios.round(3)}, Total: {sum(pca_ratios):.3f}")
