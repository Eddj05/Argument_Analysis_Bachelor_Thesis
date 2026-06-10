"""
combine_results.py — Combine three results CSV files into one

Reads results.csv, results_labels_only.csv and results_text_and_labels.csv
and combines them into a single results_combined.csv file.

Checks that all files have the same columns before combining.
Prints a summary of the combined file afterwards.

Usage:
    python combine_results.py

Input:  results.csv
        results_labels_only.csv
        results_text_and_labels.csv
Output: results_combined.csv
"""

import pandas as pd
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILES = [
    "../data/CSV/results_text_only.csv",
    "../data/CSV/results_labels_only.csv",
    "../data/CSV/results_text_and_labels.csv",
]

OUTPUT_FILE = "../data/CSV/results_combined.csv"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dfs = []

    for filepath in INPUT_FILES:
        print(f"Reading {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  {len(df)} rows, columns: {list(df.columns)}")
        dfs.append((filepath, df))

    # Check all files have the same columns
    col_sets = [set(df.columns) for _, df in dfs]
    if len(set(frozenset(c) for c in col_sets)) > 1:
        print("\nWarning: files have different columns!")
        for filepath, df in dfs:
            print(f"  {filepath}: {list(df.columns)}")
        print("Attempting to combine anyway using shared columns...")
        shared_cols = set.intersection(*col_sets)
        dfs = [(f, df[list(shared_cols)]) for f, df in dfs]
    else:
        print("\nAll files have matching columns — OK")

    # Combine
    combined = pd.concat([df for _, df in dfs], ignore_index=True)

    # Save
    combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Combined results summary")
    print(f"{'='*55}")
    print(f"  Total rows:     {len(combined)}")
    print(f"  Columns:        {list(combined.columns)}")
    print()

    if "condition" in combined.columns:
        print(f"  Rows per condition:")
        for condition, count in combined["condition"].value_counts().items():
            print(f"    {condition:<25}  {count:>7}")
        print()

    if "model" in combined.columns:
        print(f"  Rows per model:")
        for model, count in combined["model"].value_counts().items():
            print(f"    {model:<40}  {count:>7}")
        print()

    if "strategy" in combined.columns:
        print(f"  Rows per strategy:")
        for strategy, count in combined["strategy"].value_counts().items():
            print(f"    {strategy:<20}  {count:>7}")
        print()

    if "prediction" in combined.columns:
        print(f"  Prediction distribution:")
        for pred, count in combined["prediction"].value_counts().items():
            print(f"    {pred:<15}  {count:>7}  ({count/len(combined)*100:.1f}%)")
        print()

    if "delta_true" in combined.columns:
        print(f"  Delta distribution:")
        for delta, count in combined["delta_true"].value_counts().items():
            print(f"    delta={delta}  {count:>7}")
        print()

    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"  Ready to use with evaluation.py")


if __name__ == "__main__":
    main()