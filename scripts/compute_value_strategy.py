"""
convert_to_pipeline.py — Convert labeled JSON to pipeline CSV format

Reads the thresholded labeled dataset (labeled_main_dataset_t0.8.json)
and converts it to the CSV format expected by the rest of the pipeline:

    pair_id, op_text, op_values, reply_text, reply_values, delta, strategy

Each JSON pair produces TWO rows:
  - one for the winning rebuttal (delta = 1)
  - one for the losing rebuttal  (delta = 0)

The strategy column is computed using the same Schwartz-based
classification as step6_value_strategy.py.

Usage:
    python convert_to_pipeline.py

Input:  ../data/JSON/labeled_main_dataset_t0.8.json
Output: ../data/labeled_data_with_strategy.csv
"""

import json
import csv
import ast

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "../data/JSON/labeled_main_dataset_t0.8.json"
OUTPUT_FILE = "../data/CSV/labeled_data_with_strategy.csv"

# ── Schwartz Level 4 category mapping ────────────────────────────────────────

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

# ── Strategy classification ───────────────────────────────────────────────────

def get_level4(values: list) -> set:
    cats = set()
    for v in values:
        cats.update(LEVEL4_CATEGORIES.get(v.strip(), []))
    return cats


def classify_strategy(op_values: list, reply_values: list) -> str:
    op_values    = [v.strip() for v in op_values    if v.strip()]
    reply_values = [v.strip() for v in reply_values if v.strip()]

    if not op_values or not reply_values:
        return "unknown"

    # Step 1: direct value overlap → aligned
    if set(op_values) & set(reply_values):
        return "aligned"

    # Step 2: shared Level 4 category → reframing
    if get_level4(op_values) & get_level4(reply_values):
        return "reframing"

    # Step 3: nothing shared → opposition
    return "opposition"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped = 0

    for item in data:
        pair_id  = item["pair_id"]
        op_text  = item["op_context"]["text"].replace("\n", " ").strip()
        op_vals  = item["op_context"].get("predicted_values", [])

        for delta_val, rebuttal_key in [(1, "winning_rebuttal"), (0, "losing_rebuttal")]:
            rebuttal   = item[rebuttal_key]
            reply_text = rebuttal["text"].replace("\n", " ").strip()
            reply_vals = rebuttal.get("predicted_values", [])

            # Skip pairs where either side has no values
            if not op_vals or not reply_vals:
                skipped += 1
                continue

            strategy = classify_strategy(op_vals, reply_vals)

            rows.append({
                "pair_id":      pair_id,
                "op_text":      op_text,
                "op_values":    str(op_vals),
                "reply_text":   reply_text,
                "reply_values": str(reply_vals),
                "delta":        delta_val,
                "strategy":     strategy,
            })

    # Write CSV
    print(f"Writing {OUTPUT_FILE}...")
    fieldnames = ["pair_id", "op_text", "op_values", "reply_text", "reply_values", "delta", "strategy"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    strategy_counts = Counter(r["strategy"] for r in rows)
    delta_counts    = Counter(r["delta"] for r in rows)

    print(f"\nDone!")
    print(f"  Total rows:     {len(rows)}")
    print(f"  Skipped:        {skipped}  (missing values on one or both sides)")
    print()
    print(f"  Delta distribution:")
    print(f"    delta=1:      {delta_counts[1]}")
    print(f"    delta=0:      {delta_counts[0]}")
    print()
    print(f"  Strategy distribution:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"    {strategy:<15}  {count:>6}  ({pct:.1f}%)")
    print()
    print(f"  Output saved to: {OUTPUT_FILE}")
    print(f"  Ready to use with prepare_prompts.py and send_prompts.py")


if __name__ == "__main__":
    main()