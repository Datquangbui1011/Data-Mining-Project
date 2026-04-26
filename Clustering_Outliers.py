"""
=============================================================
 H-1B Visa Petitions (2011-2016) — Clustering & Outlier Detection
 CSCE 474/874  |  Group 6
=============================================================
 Inputs
 ------
   clean_full.csv   (output from Preprocessing.py)

 Outputs
 -------
   kmeans_results.csv          → original data + cluster label
   outlier_results.csv         → original data + outlier flags
   clustering_report.txt       → summary of findings
   plots/elbow_curve.png
   plots/silhouette_scores.png
   plots/kmeans_clusters_pca.png
   plots/dbscan_geo.png
   plots/outlier_iforest.png
   plots/outlier_lof.png
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 0. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Loading clean_full.csv")
print("=" * 60)

df = pd.read_csv("clean_full.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 1. FEATURE SUBSETS
# ─────────────────────────────────────────────
# Profile features for K-Means (already scaled in Preprocessing)
profile_cols = [
    "PREVAILING_WAGE", "FULL_TIME_POSITION", "YEAR",
    "LATITUDE", "LONGITUDE"
]

# All numeric + one-hot features (no target)
all_feature_cols = [c for c in df.columns if c != "CASE_STATUS_ENC"]

# Geographic features for DBSCAN
geo_cols = ["LATITUDE", "LONGITUDE"]

X_profile = df[profile_cols].values
X_all     = df[all_feature_cols].values
X_geo     = df[geo_cols].values

# ─────────────────────────────────────────────
# 2. PCA — reduce X_all for visualisation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — PCA (26 → 2 components for visualisation)")
print("=" * 60)

pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_all)
print(f"Variance explained by 2 PCs: "
      f"{pca2.explained_variance_ratio_.sum()*100:.1f}%")

pca5 = PCA(n_components=5, random_state=RANDOM_STATE)
X_pca5 = pca5.fit_transform(X_all)
print(f"Variance explained by 5 PCs: "
      f"{pca5.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. K-MEANS — Elbow + Silhouette
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — K-Means: Elbow Method & Silhouette Scores")
print("=" * 60)

# Sample for speed (elbow/silhouette on full 2.7M is slow)
SAMPLE_N = 50_000
idx = np.random.choice(len(df), SAMPLE_N, replace=False)
X_sample = X_pca5[idx]

k_range   = range(2, 11)
inertias  = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE, max_iter=300)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sample, labels))
    print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

# --- Elbow plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Curve", fontsize=13)
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores, "rs-", linewidth=2)
axes[1].set_title("Silhouette Scores vs k", fontsize=13)
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/elbow_silhouette.png", dpi=150)
plt.close()
print("Saved plots/elbow_silhouette.png")

# Pick best k by silhouette
best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nBest k by silhouette: {best_k}")

# ─────────────────────────────────────────────
# 4. K-MEANS — Fit on full dataset
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 4 — K-Means final fit (k={best_k}, full dataset)")
print("=" * 60)

km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                  random_state=RANDOM_STATE, max_iter=300)
df["CLUSTER"] = km_final.fit_predict(X_pca5)

print("Cluster distribution:")
print(df["CLUSTER"].value_counts().sort_index())

# Cluster profiles
profile_summary = df.groupby("CLUSTER")[
    ["PREVAILING_WAGE", "FULL_TIME_POSITION", "YEAR",
     "LATITUDE", "LONGITUDE", "CASE_STATUS_ENC"]
].mean().round(4)
print("\nCluster mean profiles:")
print(profile_summary.to_string())

# --- PCA scatter plot coloured by cluster ---
fig, ax = plt.subplots(figsize=(10, 7))
colors = cm.tab10(np.linspace(0, 1, best_k))
sample_pca2 = X_pca2[idx]
sample_labels = df["CLUSTER"].values[idx]

for c in range(best_k):
    mask = sample_labels == c
    ax.scatter(sample_pca2[mask, 0], sample_pca2[mask, 1],
               c=[colors[c]], label=f"Cluster {c}",
               s=5, alpha=0.4)

ax.set_title(f"K-Means Clusters (k={best_k}) — PCA 2D Projection", fontsize=13)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(markerscale=4, fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plots/kmeans_clusters_pca.png", dpi=150)
plt.close()
print("Saved plots/kmeans_clusters_pca.png")

# ─────────────────────────────────────────────
# 5. DBSCAN — Geographic Clustering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — DBSCAN Geographic Clustering (sample)")
print("=" * 60)

# DBSCAN on geo coords (already scaled 0-1)
# Sample to keep runtime manageable
GEO_SAMPLE = 30_000
geo_idx = np.random.choice(len(df), GEO_SAMPLE, replace=False)
X_geo_sample = X_geo[geo_idx]

db = DBSCAN(eps=0.03, min_samples=50, n_jobs=-1)
geo_labels = db.fit_predict(X_geo_sample)

n_clusters_db = len(set(geo_labels)) - (1 if -1 in geo_labels else 0)
n_noise_db    = (geo_labels == -1).sum()
print(f"DBSCAN clusters found : {n_clusters_db}")
print(f"Noise points (outliers): {n_noise_db:,}  "
      f"({n_noise_db/GEO_SAMPLE*100:.1f}%)")

# --- Geographic scatter ---
fig, ax = plt.subplots(figsize=(12, 7))
unique_labels = sorted(set(geo_labels))
cmap = cm.get_cmap("tab20", len(unique_labels))

for lbl in unique_labels:
    mask = geo_labels == lbl
    color = "black" if lbl == -1 else cmap(lbl)
    label = "Noise" if lbl == -1 else f"Cluster {lbl}"
    ax.scatter(X_geo_sample[mask, 1], X_geo_sample[mask, 0],
               c=[color], s=3, alpha=0.4, label=label)

ax.set_title("DBSCAN Geographic Clusters (scaled LAT/LON)", fontsize=13)
ax.set_xlabel("Longitude (scaled)")
ax.set_ylabel("Latitude (scaled)")
ax.legend(markerscale=4, fontsize=7, ncol=3, loc="lower right")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plots/dbscan_geo.png", dpi=150)
plt.close()
print("Saved plots/dbscan_geo.png")

# ─────────────────────────────────────────────
# 6. ISOLATION FOREST — Outlier Detection
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Isolation Forest Outlier Detection")
print("=" * 60)

CONTAMINATION = 0.02   # expect ~2% outliers

iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Run on profile features (full dataset — IF handles scale well)
X_outlier = df[profile_cols].values
df["IF_SCORE"]  = iso.fit_predict(X_outlier)          # -1 = outlier
df["IF_ANOMALY_SCORE"] = iso.decision_function(X_outlier)  # lower = more anomalous

n_outliers_if = (df["IF_SCORE"] == -1).sum()
print(f"Isolation Forest outliers: {n_outliers_if:,}  "
      f"({n_outliers_if/len(df)*100:.2f}%)")

print("\nOutlier wage stats vs inliers:")
print(df.groupby(df["IF_SCORE"].map({1: "Inlier", -1: "Outlier"}))
      ["PREVAILING_WAGE"].describe().round(4))

# --- Anomaly score histogram ---
fig, ax = plt.subplots(figsize=(10, 5))
inlier_scores  = df.loc[df["IF_SCORE"] ==  1, "IF_ANOMALY_SCORE"]
outlier_scores = df.loc[df["IF_SCORE"] == -1, "IF_ANOMALY_SCORE"]

ax.hist(inlier_scores,  bins=100, color="steelblue", alpha=0.6, label="Inlier")
ax.hist(outlier_scores, bins=100, color="crimson",   alpha=0.6, label="Outlier")
ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
ax.set_title("Isolation Forest — Anomaly Score Distribution", fontsize=13)
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Count")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plots/outlier_iforest.png", dpi=150)
plt.close()
print("Saved plots/outlier_iforest.png")

# ─────────────────────────────────────────────
# 7. LOCAL OUTLIER FACTOR (sample)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Local Outlier Factor (100k sample)")
print("=" * 60)

LOF_SAMPLE = 100_000
lof_idx = np.random.choice(len(df), LOF_SAMPLE, replace=False)
X_lof   = df[profile_cols].values[lof_idx]

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    n_jobs=-1
)
lof_labels = lof.fit_predict(X_lof)         # -1 = outlier
lof_scores = lof.negative_outlier_factor_   # more negative = more anomalous

n_outliers_lof = (lof_labels == -1).sum()
print(f"LOF outliers in sample: {n_outliers_lof:,}  "
      f"({n_outliers_lof/LOF_SAMPLE*100:.2f}%)")

# Map LOF results back for comparison
df["LOF_SCORE"] = np.nan
df.iloc[lof_idx, df.columns.get_loc("LOF_SCORE")] = lof_labels

# --- LOF scatter (wage vs latitude, coloured by LOF label) ---
fig, ax = plt.subplots(figsize=(10, 6))
X_lof_df = df[profile_cols].values[lof_idx]

inliers  = lof_labels == 1
outliers = lof_labels == -1

ax.scatter(X_lof_df[inliers,  0], X_lof_df[inliers,  3],
           c="steelblue", s=3, alpha=0.3, label="Inlier")
ax.scatter(X_lof_df[outliers, 0], X_lof_df[outliers, 3],
           c="crimson",   s=10, alpha=0.8, label="Outlier")

ax.set_title("LOF Outliers — Wage vs Latitude (scaled)", fontsize=13)
ax.set_xlabel("PREVAILING_WAGE (scaled)")
ax.set_ylabel("LATITUDE (scaled)")
ax.legend(markerscale=4)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("plots/outlier_lof.png", dpi=150)
plt.close()
print("Saved plots/outlier_lof.png")

# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Saving result CSVs")
print("=" * 60)

# K-Means results
kmeans_out = df[profile_cols + ["CASE_STATUS_ENC", "CLUSTER"]].copy()
kmeans_out.to_csv("kmeans_results.csv", index=False)
print(f"Saved kmeans_results.csv  — shape: {kmeans_out.shape}")

# Outlier results
outlier_out = df[profile_cols + ["CASE_STATUS_ENC",
                                  "IF_SCORE", "IF_ANOMALY_SCORE",
                                  "LOF_SCORE"]].copy()
outlier_out.to_csv("outlier_results.csv", index=False)
print(f"Saved outlier_results.csv — shape: {outlier_out.shape}")

# ─────────────────────────────────────────────
# 9. CLUSTERING REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Writing clustering_report.txt")
print("=" * 60)

report = [
    "CLUSTERING & OUTLIER DETECTION SUMMARY — H-1B (2011-2016)",
    "=" * 60,
    "",
    "DATASET",
    f"  Records used           : {len(df):,}",
    f"  Features               : {len(all_feature_cols)}",
    f"  PCA components (5)     : {pca5.explained_variance_ratio_.sum()*100:.1f}% variance explained",
    "",
    "K-MEANS CLUSTERING",
    f"  k range tested         : {list(k_range)}",
    f"  Best k (silhouette)    : {best_k}",
    f"  Silhouette scores      : {[round(s,4) for s in sil_scores]}",
    "",
    "  Cluster Sizes:",
]
for c, cnt in df["CLUSTER"].value_counts().sort_index().items():
    pct = cnt / len(df) * 100
    report.append(f"    Cluster {c}: {cnt:>10,}  ({pct:.1f}%)")

report += [
    "",
    "  Cluster Mean Profiles (key features):",
    profile_summary.to_string(),
    "",
    "DBSCAN GEOGRAPHIC CLUSTERING",
    f"  eps=0.03, min_samples=50, sample={GEO_SAMPLE:,}",
    f"  Clusters found         : {n_clusters_db}",
    f"  Noise points           : {n_noise_db:,}  ({n_noise_db/GEO_SAMPLE*100:.1f}%)",
    "",
    "OUTLIER DETECTION",
    "  Isolation Forest (full dataset):",
    f"    contamination=0.02",
    f"    Outliers flagged       : {n_outliers_if:,}  ({n_outliers_if/len(df)*100:.2f}%)",
    "",
    f"  Local Outlier Factor (sample={LOF_SAMPLE:,}):",
    f"    contamination=0.02, n_neighbors=20",
    f"    Outliers in sample     : {n_outliers_lof:,}  ({n_outliers_lof/LOF_SAMPLE*100:.2f}%)",
    "",
    "OUTPUT FILES",
    "  kmeans_results.csv     : data + CLUSTER label",
    "  outlier_results.csv    : data + IF_SCORE + LOF_SCORE",
    "  plots/elbow_silhouette.png",
    "  plots/kmeans_clusters_pca.png",
    "  plots/dbscan_geo.png",
    "  plots/outlier_iforest.png",
    "  plots/outlier_lof.png",
]

report_text = "\n".join(report)
with open("clustering_report.txt", "w") as f:
    f.write(report_text)

print(report_text)
print("\n✓ All done.")
print("  → kmeans_results.csv")
print("  → outlier_results.csv")
print("  → clustering_report.txt")
print("  → plots/")
