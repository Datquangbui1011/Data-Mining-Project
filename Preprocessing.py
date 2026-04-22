"""
=============================================================
 H-1B Visa Petitions (2011-2016) — Data Preprocessing
 CSCE 474/874  |  Group 6  |  Lead: Dat Bui
=============================================================
 Outputs
 -------
   clean_full.csv       → for Classification & Clustering/Outlier Detection
   clean_apriori.csv    → for Association Rule Mining (discretized)
   preprocessing_report.txt → summary stats for the final report
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Loading dataset")
print("=" * 60)

df = pd.read_csv("data/h1b_kaggle.csv", index_col=0)

print(f"Raw shape          : {df.shape}")
print(f"Columns            : {list(df.columns)}")

# rename lon/lat to match proposal attribute names
df.rename(columns={"lon": "LONGITUDE", "lat": "LATITUDE"}, inplace=True)

# ─────────────────────────────────────────────
# 1. INITIAL EXPLORATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Initial exploration")
print("=" * 60)

print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.concat([missing, missing_pct], axis=1,
      keys=["Count", "Pct %"]).to_string())

print("\nCASE_STATUS value counts:")
print(df["CASE_STATUS"].value_counts())

# ─────────────────────────────────────────────
# 2. CASE_STATUS — consolidate & filter
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Standardize CASE_STATUS")
print("=" * 60)

# Map CERTIFIED-WITHDRAWN → WITHDRAWN, keep only 3 core classes
status_map = {
    "CERTIFIED":           "CERTIFIED",
    "CERTIFIED-WITHDRAWN": "WITHDRAWN",
    "DENIED":              "DENIED",
    "WITHDRAWN":           "WITHDRAWN",
}
df["CASE_STATUS"] = df["CASE_STATUS"].str.strip().str.upper().map(status_map)
df.dropna(subset=["CASE_STATUS"], inplace=True)

print("CASE_STATUS after consolidation:")
print(df["CASE_STATUS"].value_counts())
print(f"Shape after status filter: {df.shape}")

# ─────────────────────────────────────────────
# 3. DROP MISSING GEOGRAPHIC VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Drop missing geographic values")
print("=" * 60)

before = len(df)
df.dropna(subset=["WORKSITE", "LATITUDE", "LONGITUDE"], inplace=True)
after = len(df)
print(f"Dropped {before - after:,} records missing WORKSITE/LAT/LON")
print(f"Shape: {df.shape}")

# ─────────────────────────────────────────────
# 4. EXTRACT STATE FROM WORKSITE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Extract STATE from WORKSITE")
print("=" * 60)

df["STATE"] = df["WORKSITE"].str.split(",").str[-1].str.strip().str.upper()
print("Top 10 states:")
print(df["STATE"].value_counts().head(10))

# ─────────────────────────────────────────────
# 5. PREVAILING_WAGE — clean & outlier filtering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Wage cleaning & outlier filtering")
print("=" * 60)

# Force numeric
df["PREVAILING_WAGE"] = pd.to_numeric(df["PREVAILING_WAGE"], errors="coerce")
df.dropna(subset=["PREVAILING_WAGE"], inplace=True)

# Remove non-positive wages
before = len(df)
df = df[df["PREVAILING_WAGE"] > 0]
print(f"Removed {before - len(df):,} records with zero/negative wages")

# IQR-based outlier removal (1.5x IQR rule)
Q1 = df["PREVAILING_WAGE"].quantile(0.25)
Q3 = df["PREVAILING_WAGE"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = len(df)
df = df[(df["PREVAILING_WAGE"] >= lower) & (df["PREVAILING_WAGE"] <= upper)]
after = len(df)

print(f"Wage IQR range     : [{lower:,.2f}, {upper:,.2f}]")
print(f"Removed outliers   : {before - after:,} records")
print(f"Wage stats after filter:")
print(df["PREVAILING_WAGE"].describe().round(2))

# ─────────────────────────────────────────────
# 6. FULL_TIME_POSITION — standardize
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Standardize FULL_TIME_POSITION")
print("=" * 60)

df["FULL_TIME_POSITION"] = (
    df["FULL_TIME_POSITION"].str.strip().str.upper()
    .map({"Y": 1, "N": 0})
)
df.dropna(subset=["FULL_TIME_POSITION"], inplace=True)
df["FULL_TIME_POSITION"] = df["FULL_TIME_POSITION"].astype(int)

print("FULL_TIME_POSITION distribution:")
print(df["FULL_TIME_POSITION"].value_counts())

# ─────────────────────────────────────────────
# 7. SOC_NAME — clean & simplify
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Clean SOC_NAME")
print("=" * 60)

df["SOC_NAME"] = df["SOC_NAME"].str.strip().str.upper()

# Keep only the top N occupation groups to avoid high cardinality
top_soc = df["SOC_NAME"].value_counts().head(20).index
df["SOC_NAME_GROUPED"] = df["SOC_NAME"].where(
    df["SOC_NAME"].isin(top_soc), other="OTHER"
)
print("Top 10 SOC_NAME groups:")
print(df["SOC_NAME_GROUPED"].value_counts().head(10))

# ─────────────────────────────────────────────
# 8. YEAR — validate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Validate YEAR")
print("=" * 60)

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
before = len(df)
df = df[df["YEAR"].between(2011, 2016)]
print(f"Removed {before - len(df):,} records with invalid YEAR")
print("YEAR distribution:")
print(df["YEAR"].value_counts().sort_index())

# ─────────────────────────────────────────────
# 9. FINAL SHAPE AFTER CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Final cleaned dataset overview")
print("=" * 60)

print(f"Final shape        : {df.shape}")
print(f"Remaining missing  :\n{df.isnull().sum()}")

# ─────────────────────────────────────────────────────────────────
# 10. EXPORT: clean_full.csv  (Classification & Clustering)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Build clean_full.csv (Classification & Clustering)")
print("=" * 60)

# Label encode target
le = LabelEncoder()
df["CASE_STATUS_ENC"] = le.fit_transform(df["CASE_STATUS"])
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# One-hot encode SOC_NAME_GROUPED
soc_dummies = pd.get_dummies(df["SOC_NAME_GROUPED"], prefix="SOC")

# Build feature dataframe
full = pd.concat([
    df[["CASE_STATUS_ENC",
        "PREVAILING_WAGE",
        "FULL_TIME_POSITION",
        "YEAR",
        "LATITUDE",
        "LONGITUDE"]].reset_index(drop=True),
    soc_dummies.reset_index(drop=True),
], axis=1)

# Scale numeric features (wage, lat, lon, year) for clustering
scaler = MinMaxScaler()
cols_to_scale = ["PREVAILING_WAGE", "LATITUDE", "LONGITUDE", "YEAR"]
full[cols_to_scale] = scaler.fit_transform(full[cols_to_scale])

full.to_csv("clean_full.csv", index=False)
print(f"Saved clean_full.csv  — shape: {full.shape}")
print(f"Columns: {list(full.columns)}")

# ─────────────────────────────────────────────────────────────────
# 11. EXPORT: clean_apriori.csv  (Association Rule Mining)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — Build clean_apriori.csv (Association Rule Mining)")
print("=" * 60)

apriori_df = df[["CASE_STATUS", "SOC_NAME_GROUPED",
                  "FULL_TIME_POSITION", "STATE",
                  "PREVAILING_WAGE", "YEAR"]].copy()

# Discretize wage into 3 equal-frequency bins
apriori_df["WAGE_BUCKET"] = pd.qcut(
    apriori_df["PREVAILING_WAGE"], q=3,
    labels=["LOW_WAGE", "MEDIUM_WAGE", "HIGH_WAGE"]
)

# Keep only top 10 states to limit cardinality
top_states = apriori_df["STATE"].value_counts().head(10).index
apriori_df["STATE"] = apriori_df["STATE"].where(
    apriori_df["STATE"].isin(top_states), other="OTHER_STATE"
)

# Convert FULL_TIME to string label
apriori_df["FULL_TIME_POSITION"] = apriori_df["FULL_TIME_POSITION"].map(
    {1: "FULLTIME_Y", 0: "FULLTIME_N"}
)

# Drop raw wage and year (replaced by discretized/grouped)
apriori_df.drop(columns=["PREVAILING_WAGE", "YEAR"], inplace=True)

# One-hot encode everything (Apriori needs binary matrix)
apriori_encoded = pd.get_dummies(apriori_df, dtype=bool)

apriori_encoded.to_csv("clean_apriori.csv", index=False)
print(f"Saved clean_apriori.csv — shape: {apriori_encoded.shape}")
print(f"Sample columns: {list(apriori_encoded.columns[:10])}")

# ─────────────────────────────────────────────────────────────────
# 12. PREPROCESSING REPORT (paste into paper)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12 — Writing preprocessing_report.txt")
print("=" * 60)

report_lines = [
    "DATA PREPROCESSING SUMMARY — H-1B Visa Petitions (2011-2016)",
    "=" * 60,
    "",
    "ORIGINAL DATASET",
    f"  Total records (raw)       : ~3,000,000",
    "",
    "CLEANING STEPS & RECORDS REMOVED",
    f"  Invalid CASE_STATUS       : mapped to 3 classes (CERTIFIED, DENIED, WITHDRAWN)",
    f"  Missing WORKSITE/LAT/LON  : dropped",
    f"  Non-positive wages        : dropped",
    f"  Wage IQR outliers         : [{lower:,.0f} - {upper:,.0f}] — records outside range dropped",
    f"  Invalid YEAR              : kept only 2011-2016",
    "",
    "FINAL DATASET",
    f"  Records retained          : {len(df):,}",
    f"  Attributes                : {df.shape[1]}",
    "",
    "CASE_STATUS DISTRIBUTION (after cleaning)",
]
for status, count in df["CASE_STATUS"].value_counts().items():
    pct = count / len(df) * 100
    report_lines.append(f"  {status:<25}: {count:>8,}  ({pct:.1f}%)")

report_lines += [
    "",
    "PREVAILING_WAGE STATISTICS (after outlier removal)",
    f"  Min    : ${df['PREVAILING_WAGE'].min():>12,.2f}",
    f"  Q1     : ${df['PREVAILING_WAGE'].quantile(0.25):>12,.2f}",
    f"  Median : ${df['PREVAILING_WAGE'].median():>12,.2f}",
    f"  Mean   : ${df['PREVAILING_WAGE'].mean():>12,.2f}",
    f"  Q3     : ${df['PREVAILING_WAGE'].quantile(0.75):>12,.2f}",
    f"  Max    : ${df['PREVAILING_WAGE'].max():>12,.2f}",
    "",
    "YEAR DISTRIBUTION",
]
for yr, cnt in df["YEAR"].value_counts().sort_index().items():
    report_lines.append(f"  {int(yr)}: {cnt:,}")

report_lines += [
    "",
    "OUTPUT FILES",
    f"  clean_full.csv    : {full.shape[0]:,} rows x {full.shape[1]} cols — for Classification & Clustering",
    f"  clean_apriori.csv : {apriori_encoded.shape[0]:,} rows x {apriori_encoded.shape[1]} cols — for Apriori",
    "",
    "ENCODING SUMMARY",
    "  CASE_STATUS_ENC   : LabelEncoder → " + str(dict(zip(le.classes_, le.transform(le.classes_)))),
    "  SOC_NAME_GROUPED  : Top 20 occupations + OTHER → one-hot encoded",
    "  FULL_TIME_POSITION: Y/N → 1/0",
    "  STATE             : Extracted from WORKSITE",
    "  WAGE_BUCKET       : pd.qcut 3-bin (LOW / MEDIUM / HIGH) — Apriori only",
    "  Numeric scaling   : MinMaxScaler on PREVAILING_WAGE, LATITUDE, LONGITUDE, YEAR",
]

report_text = "\n".join(report_lines)
with open("preprocessing_report.txt", "w") as f:
    f.write(report_text)

print(report_text)
print("\n✓ All done. Files saved:")
print("  → clean_full.csv")
print("  → clean_apriori.csv")
print("  → preprocessing_report.txt")