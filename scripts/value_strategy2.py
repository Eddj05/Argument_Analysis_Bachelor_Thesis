"""
Step 6 — Compute Value Strategy per Pair (Level 4 category version)

Strategy definitions:
  - Aligned:    at least one overlapping value label
  - Reframing:  no overlap, but reply shares at least one Level 4 category with OP
  - Opposition: no overlap, and no shared Level 4 category

Note: some values belong to two Level 4 categories (e.g. Hedonism belongs to
both Openness to change and Self-enhancement). This is handled by giving each
value a list of categories rather than a single one.

Usage:
    python step6_value_strategy.py

Input:  labeled_data_example.csv  (change INPUT_FILE below)
Output: labeled_data_with_strategy.csv
"""

import ast
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "../data/CSV/labeled_data_example.csv"
OUTPUT_FILE = "../data/CSV/labeled_data_with_strategy2.csv"

# ── Level 4 category mapping ──────────────────────────────────────────────────
# Based on: https://touche.webis.de/clef24/touche24-web/human-value-detection.html
# Values on the boundary between two categories are assigned to both.

LEVEL4_CATEGORIES = {
    # Openness to change
    "Self-direction: thought":    ["Openness to change"],
    "Self-direction: action":     ["Openness to change"],
    "Stimulation":                ["Openness to change"],
    "Hedonism":                   ["Openness to change", "Self-enhancement"],

    # Self-enhancement
    "Achievement":                ["Self-enhancement"],
    "Power: dominance":           ["Self-enhancement"],
    "Power: resources":           ["Self-enhancement"],
    "Face":                       ["Self-enhancement", "Conservation"],

    # Conservation
    "Security: personal":         ["Conservation"],
    "Security: societal":         ["Conservation"],
    "Tradition":                  ["Conservation"],
    "Conformity: rules":          ["Conservation"],
    "Conformity: interpersonal":  ["Conservation"],
    "Humility":                   ["Conservation", "Self-transcendence"],

    # Self-transcendence
    "Benevolence: caring":        ["Self-transcendence"],
    "Benevolence: dependability": ["Self-transcendence"],
    "Universalism: concern":      ["Self-transcendence"],
    "Universalism: nature":       ["Self-transcendence"],
    "Universalism: tolerance":    ["Self-transcendence"],
    "Universalism: objectivity":  ["Self-transcendence"],
}

# ── Helper functions ──────────────────────────────────────────────────────────

def get_level4_categories(values: list) -> set:
    categories = set()
    for v in values:
        cats = LEVEL4_CATEGORIES.get(v.strip(), [])
        categories.update(cats)
    return categories


def classify_strategy(op_values: list, reply_values: list) -> str:
    op_values    = [v.strip() for v in op_values    if v.strip()]
    reply_values = [v.strip() for v in reply_values if v.strip()]

    if not op_values or not reply_values:
        return "unknown"

    # Step 1: direct value overlap → aligned
    if set(op_values) & set(reply_values):
        return "aligned"

    # Step 2: shared Level 4 category → reframing
    op_cats    = get_level4_categories(op_values)
    reply_cats = get_level4_categories(reply_values)

    if not op_cats or not reply_cats:
        return "unknown"

    if op_cats & reply_cats:
        return "reframing"

    # Step 3: nothing shared → opposition
    return "opposition"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    def parse_values(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return [v.strip() for v in str(val).split(",") if v.strip()]

    df["op_values"]    = df["op_values"].apply(parse_values)
    df["reply_values"] = df["reply_values"].apply(parse_values)

    df["strategy"] = df.apply(
        lambda row: classify_strategy(row["op_values"], row["reply_values"]),
        axis=1
    )

    print(f"\nDone! Processed {len(df)} rows.\n")
    print("Strategy distribution:")
    print(df["strategy"].value_counts().to_string())
    print()


    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()