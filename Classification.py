"""
=============================================================
 H-1B Visa Petitions (2011-2016) — Objective 3: Classification
 CSCE 474/874  |  Group 6  |  Lead: Thang Do
=============================================================
 Models  : Decision Tree, Random Forest, K-Nearest Neighbors
 Outputs :
   fig2_confusion_matrix.png     → best model confusion matrix
   fig3_feature_importance.png   → Random Forest top-10 features
   fig_model_comparison.png      → side-by-side metric bar chart
   classification_report.txt     → full results for the paper
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

CLASS_NAMES = ["CERTIFIED", "DENIED", "WITHDRAWN"]
LABEL_MAP   = {0: "CERTIFIED", 1: "DENIED", 2: "WITHDRAWN"}
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 0. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Loading clean_full.csv")
print("=" * 60)

df = pd.read_csv("clean_full.csv")
print(f"Shape: {df.shape}")

TARGET   = "CASE_STATUS_ENC"
FEATURES = [c for c in df.columns if c != TARGET]

X_full = df[FEATURES].values
y_full = df[TARGET].values

print("\nClass distribution (full):")
for enc, name in LABEL_MAP.items():
    n = (y_full == enc).sum()
    print(f"  {name} ({enc}): {n:,}  ({n/len(y_full)*100:.1f}%)")

# ─────────────────────────────────────────────
# 1. STRATIFIED SAMPLE
# ─────────────────────────────────────────────
# DT / RF  — 300 K rows (keeps class proportions, tractable CV)
# KNN      — 60 K rows  (KNN is O(n) at prediction time)
print("\n" + "=" * 60)
print("STEP 1 — Stratified sampling")
print("=" * 60)

X_drf, _, y_drf, _ = train_test_split(
    X_full, y_full, train_size=300_000, stratify=y_full, random_state=RANDOM_STATE
)
X_knn_all, _, y_knn_all, _ = train_test_split(
    X_full, y_full, train_size=60_000,  stratify=y_full, random_state=RANDOM_STATE
)
print(f"DT/RF sample : {X_drf.shape[0]:,}")
print(f"KNN  sample  : {X_knn_all.shape[0]:,}")

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLITS  (80/20)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Train/Test splits (80/20 stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_drf, y_drf, test_size=0.2, stratify=y_drf, random_state=RANDOM_STATE
)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn_all, y_knn_all, test_size=0.2, stratify=y_knn_all, random_state=RANDOM_STATE
)
print(f"DT/RF  Train {X_train.shape}  Test {X_test.shape}")
print(f"KNN    Train {X_train_knn.shape}  Test {X_test_knn.shape}")

# ─────────────────────────────────────────────
# 2b. BALANCED TRAINING SET FOR KNN
#     (KNN has no class_weight — use random undersampling)
# ─────────────────────────────────────────────
print("\n-- Balancing KNN training set via undersampling --")

frames_X, frames_y = [], []
min_class_count = min((y_train_knn == c).sum() for c in np.unique(y_train_knn))
for c in np.unique(y_train_knn):
    idx = np.where(y_train_knn == c)[0]
    chosen = np.random.default_rng(RANDOM_STATE).choice(idx, size=min_class_count, replace=False)
    frames_X.append(X_train_knn[chosen])
    frames_y.append(y_train_knn[chosen])

X_train_knn_bal = np.vstack(frames_X)
y_train_knn_bal = np.concatenate(frames_y)
print(f"Balanced KNN train: {X_train_knn_bal.shape[0]:,} "
      f"({min_class_count:,} per class × 3 classes)")

# ─────────────────────────────────────────────
# 3. CROSS-VALIDATION SETUP
#    Use a 100 K subset of the DT/RF training data for CV speed
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_cv, _, y_cv, _ = train_test_split(
    X_train, y_train, train_size=100_000, stratify=y_train, random_state=RANDOM_STATE
)

# ─────────────────────────────────────────────
# 4a. DECISION TREE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4a — Decision Tree (class_weight='balanced')")
print("=" * 60)

dt = DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

dt_metrics = {
    "accuracy" : accuracy_score(y_test, y_pred_dt),
    "precision": precision_score(y_test, y_pred_dt, average="weighted", zero_division=0),
    "recall"   : recall_score(y_test, y_pred_dt, average="weighted"),
    "f1"       : f1_score(y_test, y_pred_dt, average="weighted"),
}
cv_dt = cross_val_score(dt, X_cv, y_cv, cv=cv, scoring="f1_weighted", n_jobs=-1)
dt_metrics["cv_mean"] = cv_dt.mean()
dt_metrics["cv_std"]  = cv_dt.std()

print(f"  Accuracy : {dt_metrics['accuracy']:.4f}")
print(f"  Precision: {dt_metrics['precision']:.4f}")
print(f"  Recall   : {dt_metrics['recall']:.4f}")
print(f"  F1-Score : {dt_metrics['f1']:.4f}")
print(f"  5-Fold CV F1: {dt_metrics['cv_mean']:.4f} ± {dt_metrics['cv_std']:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred_dt, target_names=CLASS_NAMES, zero_division=0))

# ─────────────────────────────────────────────
# 4b. RANDOM FOREST
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4b — Random Forest (class_weight='balanced')")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_metrics = {
    "accuracy" : accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf, average="weighted", zero_division=0),
    "recall"   : recall_score(y_test, y_pred_rf, average="weighted"),
    "f1"       : f1_score(y_test, y_pred_rf, average="weighted"),
}
cv_rf = cross_val_score(rf, X_cv, y_cv, cv=cv, scoring="f1_weighted", n_jobs=-1)
rf_metrics["cv_mean"] = cv_rf.mean()
rf_metrics["cv_std"]  = cv_rf.std()

print(f"  Accuracy : {rf_metrics['accuracy']:.4f}")
print(f"  Precision: {rf_metrics['precision']:.4f}")
print(f"  Recall   : {rf_metrics['recall']:.4f}")
print(f"  F1-Score : {rf_metrics['f1']:.4f}")
print(f"  5-Fold CV F1: {rf_metrics['cv_mean']:.4f} ± {rf_metrics['cv_std']:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, zero_division=0))

# Feature importances
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(feat_imp.head(10).to_string())

# ─────────────────────────────────────────────
# 4c. K-NEAREST NEIGHBORS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4c — KNN k=7 (balanced undersampling for training)")
print("=" * 60)

knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1)
knn.fit(X_train_knn_bal, y_train_knn_bal)
y_pred_knn = knn.predict(X_test_knn)

knn_metrics = {
    "accuracy" : accuracy_score(y_test_knn, y_pred_knn),
    "precision": precision_score(y_test_knn, y_pred_knn, average="weighted", zero_division=0),
    "recall"   : recall_score(y_test_knn, y_pred_knn, average="weighted"),
    "f1"       : f1_score(y_test_knn, y_pred_knn, average="weighted"),
}
cv_knn = cross_val_score(knn, X_train_knn_bal, y_train_knn_bal, cv=cv, scoring="f1_weighted", n_jobs=-1)
knn_metrics["cv_mean"] = cv_knn.mean()
knn_metrics["cv_std"]  = cv_knn.std()

print(f"  Accuracy : {knn_metrics['accuracy']:.4f}")
print(f"  Precision: {knn_metrics['precision']:.4f}")
print(f"  Recall   : {knn_metrics['recall']:.4f}")
print(f"  F1-Score : {knn_metrics['f1']:.4f}")
print(f"  5-Fold CV F1: {knn_metrics['cv_mean']:.4f} ± {knn_metrics['cv_std']:.4f}")
print("\nFull classification report:")
print(classification_report(y_test_knn, y_pred_knn, target_names=CLASS_NAMES, zero_division=0))

# ─────────────────────────────────────────────
# 5. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Summary comparison table")
print("=" * 60)

all_results = {
    "Decision Tree" : (dt_metrics,  y_test,     y_pred_dt),
    "Random Forest" : (rf_metrics,  y_test,     y_pred_rf),
    "KNN"           : (knn_metrics, y_test_knn, y_pred_knn),
}

header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV F1 (5-fold)':>20}"
sep    = "-" * len(header)
print(header)
print(sep)
for name, (m, _, __) in all_results.items():
    cv_str = f"{m['cv_mean']:.4f}±{m['cv_std']:.4f}"
    print(f"{name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>10.4f} {m['f1']:>10.4f} {cv_str:>20}")

best_name = max(all_results, key=lambda n: all_results[n][0]["f1"])
print(f"\nBest model by weighted F1: {best_name}")

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Saving figures")
print("=" * 60)

# --- Fig 2: Confusion matrix (best model) ---
best_m, best_yt, best_yp = all_results[best_name]
cm = confusion_matrix(best_yt, best_yp)
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_confusion_matrix.png", dpi=150)
plt.close()
print("Saved fig2_confusion_matrix.png")

# --- Fig 3: Random Forest feature importances (top 10) ---
top10 = feat_imp.head(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2196F3"] * len(top10)
top10.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_title("Top 10 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5)
for i, (val, label) in enumerate(zip(top10.values, top10.index)):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("fig3_feature_importance.png", dpi=150)
plt.close()
print("Saved fig3_feature_importance.png")

# --- Fig: Model comparison bar chart ---
model_names  = list(all_results.keys())
metric_keys  = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
x     = np.arange(len(model_names))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5))
for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
    vals = [all_results[m][0][key] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=label)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.50, 1.02)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Classification Model Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig_model_comparison.png", dpi=150)
plt.close()
print("Saved fig_model_comparison.png")

# ─────────────────────────────────────────────
# 7. WRITE TEXT REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Writing classification_report.txt")
print("=" * 60)

lines = [
    "CLASSIFICATION RESULTS — H-1B Visa Petitions (2011-2016)",
    "=" * 60,
    "",
    f"Dataset size (full)       : {len(y_full):,} records",
    f"Sample size (DT/RF)       : {X_drf.shape[0]:,}",
    f"Sample size (KNN)         : {X_knn_all.shape[0]:,}",
    f"Train/Test split          : 80/20 (stratified)",
    f"CV                        : 5-fold stratified",
    "",
    "CLASS DISTRIBUTION (full dataset)",
]
for enc, name in LABEL_MAP.items():
    n = (y_full == enc).sum()
    lines.append(f"  {name:<12}: {n:>10,}  ({n/len(y_full)*100:.1f}%)")

lines += [
    "",
    "IMBALANCE STRATEGY",
    "  Decision Tree  : class_weight='balanced'",
    "  Random Forest  : class_weight='balanced'",
    "  KNN            : random undersampling to minority class size",
    "",
    "MODEL COMPARISON",
    f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV F1':>15}",
    "  " + "-" * 75,
]
for name, (m, _, __) in all_results.items():
    cv_str = f"{m['cv_mean']:.4f}±{m['cv_std']:.4f}"
    lines.append(
        f"  {name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
        f"{m['recall']:>10.4f} {m['f1']:>10.4f} {cv_str:>15}"
    )

lines += [
    "",
    f"Best model (weighted F1)  : {best_name}",
    "",
    "=" * 60,
    "DECISION TREE — Full Classification Report",
    "=" * 60,
    classification_report(y_test, y_pred_dt, target_names=CLASS_NAMES, zero_division=0),
    "=" * 60,
    "RANDOM FOREST — Full Classification Report",
    "=" * 60,
    classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, zero_division=0),
    "=" * 60,
    "KNN — Full Classification Report",
    "=" * 60,
    classification_report(y_test_knn, y_pred_knn, target_names=CLASS_NAMES, zero_division=0),
    "",
    "RANDOM FOREST — Top 10 Feature Importances",
    "=" * 60,
]
for feat, imp in feat_imp.head(10).items():
    lines.append(f"  {feat:<50} {imp:.6f}")

report_text = "\n".join(lines)
with open("classification_report.txt", "w") as f:
    f.write(report_text)

print(report_text)
print("\n✓ Classification complete. Output files:")
print("  → classification_report.txt")
print("  → fig2_confusion_matrix.png")
print("  → fig3_feature_importance.png")
print("  → fig_model_comparison.png")
