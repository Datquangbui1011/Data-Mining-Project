"""
Association Rule Mining for H-1B Visa Petitions
CSCE 474/874 | Group 6
Lead: Komlan Akakpo

Input:
    clean_apriori.csv

Outputs:
    association_results/
        frequent_itemsets_*.csv
        association_rules_*.csv
        outcome_rules_*.csv
        top_rules_*.csv
        top_outcome_rules_*.csv
        association_rule_summary.txt

Purpose:
    Discover frequent patterns among H-1B petition attributes using Apriori.
    Also identify outcome-focused rules where the consequent includes:
        CASE_STATUS_CERTIFIED
        CASE_STATUS_DENIED
        CASE_STATUS_WITHDRAWN

Note:
    The full clean_apriori.csv file has 2,785,695 rows x 40 columns.
    Running Apriori on the full dataset exceeded local memory limits on Mac,
    so this script uses a random sample of 300,000 rows.
"""

import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

INPUT_FILE = "clean_apriori.csv"
OUTPUT_DIR = "association_results"

# Use a random sample to avoid memory issues with full Apriori.
# Increase to 500000 later if your computer can handle it.
# Set to None only on a stronger computer.
SAMPLE_SIZE = 300000

# Final cleaned Apriori dataset size from preprocessing report.
TOTAL_ROWS = 2785695

RANDOM_STATE = 42
CHUNK_SIZE = 100000

THRESHOLD_CONFIGS = [
    {"min_support": 0.01, "min_confidence": 0.60},
    {"min_support": 0.05, "min_confidence": 0.70},
    {"min_support": 0.10, "min_confidence": 0.80},
]

OUTCOME_COLUMNS = {
    "CASE_STATUS_CERTIFIED",
    "CASE_STATUS_DENIED",
    "CASE_STATUS_WITHDRAWN",
}


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def ensure_boolean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Apriori input dataframe to true boolean values.

    This is important because CSV values may be read as strings.
    Directly using astype(bool) is unsafe because the string 'False'
    can incorrectly become True.
    """
    bool_map = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }

    for col in df.columns:
        if df[col].dtype != bool:
            df[col] = df[col].map(bool_map)

    if df.isnull().any().any():
        bad_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Some columns could not be converted to boolean values: {bad_cols}"
        )

    return df.astype(bool)


def itemset_to_string(itemset) -> str:
    """
    Convert a frozenset itemset into a readable comma-separated string.
    """
    return ", ".join(sorted(list(itemset)))


def save_top_rules(rules: pd.DataFrame, path: str, top_n: int = 20) -> None:
    """
    Save simplified top rules for easier report writing.
    """
    columns_to_save = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
    ]

    if rules.empty:
        empty_df = pd.DataFrame(columns=columns_to_save)
        empty_df.to_csv(path, index=False)
        return

    simple_rules = rules.head(top_n).copy()
    simple_rules["antecedents"] = simple_rules["antecedents"].apply(itemset_to_string)
    simple_rules["consequents"] = simple_rules["consequents"].apply(itemset_to_string)

    simple_rules[columns_to_save].to_csv(path, index=False)


def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Load either the full dataset or a random sample.

    For SAMPLE_SIZE = None:
        Reads the full clean_apriori.csv file.

    For SAMPLE_SIZE = number:
        Reads the file in chunks and randomly samples from each chunk.
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Make sure clean_apriori.csv is "
            f"in the same folder as this script."
        )

    if SAMPLE_SIZE is None:
        print("Loading full dataset...")
        df = pd.read_csv(INPUT_FILE)
        dataset_note = "FULL DATASET"
        return df, dataset_note

    print(f"Creating random sample of {SAMPLE_SIZE:,} rows...")
    sample_fraction = SAMPLE_SIZE / TOTAL_ROWS

    sampled_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
        start=1
    ):
        sampled_chunk = chunk.sample(
            frac=sample_fraction,
            random_state=RANDOM_STATE + chunk_number
        )
        sampled_chunks.append(sampled_chunk)

    df = pd.concat(sampled_chunks, ignore_index=True)

    # Because chunk-based fractional sampling can produce slightly more
    # than SAMPLE_SIZE rows, trim to the exact target size.
    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    dataset_note = f"RANDOM SAMPLE OF {len(df):,} ROWS"
    return df, dataset_note


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Association Rule Mining - H-1B Visa Petitions")
    print("=" * 80)

    print(f"Input file: {INPUT_FILE}")
    df, dataset_note = load_dataset()

    print(f"Dataset loaded: {df.shape}")
    print(f"Mode: {dataset_note}")

    print("Converting data to boolean format...")
    df = ensure_boolean_dataframe(df)

    summary_lines = []
    summary_lines.append("ASSOCIATION RULE MINING SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Input file: {INPUT_FILE}")
    summary_lines.append(f"Dataset used: {dataset_note}")
    summary_lines.append(f"Shape used: {df.shape[0]:,} rows x {df.shape[1]} columns")
    summary_lines.append("")
    summary_lines.append(
        "Note: A random sample was used because Apriori on the full "
        "2,785,695-row dataset exceeded local memory limits."
    )
    summary_lines.append("")
    summary_lines.append("Threshold Experiments")
    summary_lines.append("-" * 80)

    for config in THRESHOLD_CONFIGS:
        min_support = config["min_support"]
        min_confidence = config["min_confidence"]

        print("\n" + "-" * 80)
        print(f"Running Apriori: support={min_support}, confidence={min_confidence}")
        print("-" * 80)

        frequent_itemsets = apriori(
            df,
            min_support=min_support,
            use_colnames=True
        )

        print(f"Frequent itemsets found: {len(frequent_itemsets)}")

        support_label = str(min_support).replace(".", "")
        confidence_label = str(min_confidence).replace(".", "")

        itemset_file = (
            f"{OUTPUT_DIR}/frequent_itemsets_s{support_label}_c{confidence_label}.csv"
        )
        rules_file = (
            f"{OUTPUT_DIR}/association_rules_s{support_label}_c{confidence_label}.csv"
        )
        outcome_rules_file = (
            f"{OUTPUT_DIR}/outcome_rules_s{support_label}_c{confidence_label}.csv"
        )
        top_rules_file = (
            f"{OUTPUT_DIR}/top_rules_s{support_label}_c{confidence_label}.csv"
        )
        top_outcome_rules_file = (
            f"{OUTPUT_DIR}/top_outcome_rules_s{support_label}_c{confidence_label}.csv"
        )

        frequent_itemsets.to_csv(itemset_file, index=False)

        if frequent_itemsets.empty:
            print("No frequent itemsets found for this threshold.")

            empty_rules = pd.DataFrame()
            empty_rules.to_csv(rules_file, index=False)
            empty_rules.to_csv(outcome_rules_file, index=False)

            save_top_rules(empty_rules, top_rules_file)
            save_top_rules(empty_rules, top_outcome_rules_file)

            summary_lines.append(
                f"Support={min_support}, Confidence={min_confidence}: "
                f"0 itemsets, 0 total rules, 0 outcome-focused rules"
            )
            summary_lines.append("")
            continue

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        if not rules.empty:
            rules = rules.sort_values(
                by=["lift", "confidence"],
                ascending=False
            )

        # Outcome-focused rules:
        # Keep only rules where the consequent includes a case status.
        if rules.empty:
            outcome_rules = rules.copy()
        else:
            outcome_rules = rules[
                rules["consequents"].apply(
                    lambda x: any(item in OUTCOME_COLUMNS for item in x)
                )
            ].copy()

            if not outcome_rules.empty:
                outcome_rules = outcome_rules.sort_values(
                    by=["lift", "confidence"],
                    ascending=False
                )

        print(f"Total rules generated: {len(rules)}")
        print(f"Outcome-focused rules: {len(outcome_rules)}")

        rules.to_csv(rules_file, index=False)
        outcome_rules.to_csv(outcome_rules_file, index=False)

        save_top_rules(rules, top_rules_file, top_n=20)
        save_top_rules(outcome_rules, top_outcome_rules_file, top_n=20)

        summary_lines.append(
            f"Support={min_support}, Confidence={min_confidence}: "
            f"{len(frequent_itemsets)} itemsets, "
            f"{len(rules)} total rules, "
            f"{len(outcome_rules)} outcome-focused rules"
        )

        # Top general rules
        summary_lines.append("")
        summary_lines.append(
            f"Top 5 General Rules "
            f"(support={min_support}, confidence={min_confidence})"
        )

        if rules.empty:
            summary_lines.append("No rules generated.")
        else:
            for i, (_, row) in enumerate(rules.head(5).iterrows(), start=1):
                antecedents = itemset_to_string(row["antecedents"])
                consequents = itemset_to_string(row["consequents"])

                summary_lines.append(
                    f"{i}. IF {antecedents} THEN {consequents} | "
                    f"support={row['support']:.4f}, "
                    f"confidence={row['confidence']:.4f}, "
                    f"lift={row['lift']:.4f}"
                )

        # Top outcome-focused rules
        summary_lines.append("")
        summary_lines.append(
            f"Top 5 Outcome-Focused Rules "
            f"(support={min_support}, confidence={min_confidence})"
        )

        if outcome_rules.empty:
            summary_lines.append("No outcome-focused rules generated.")
        else:
            for i, (_, row) in enumerate(outcome_rules.head(5).iterrows(), start=1):
                antecedents = itemset_to_string(row["antecedents"])
                consequents = itemset_to_string(row["consequents"])

                summary_lines.append(
                    f"{i}. IF {antecedents} THEN {consequents} | "
                    f"support={row['support']:.4f}, "
                    f"confidence={row['confidence']:.4f}, "
                    f"lift={row['lift']:.4f}"
                )

        summary_lines.append("")

    summary_path = f"{OUTPUT_DIR}/association_rule_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "=" * 80)
    print("Association rule mining complete.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Summary file: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()