"""
apply_threshold.py — Apply confidence threshold to labeled dataset

Reads labeled_main_dataset.json and creates a cleaned version where
only value labels with a predicted probability >= THRESHOLD are kept.

Run with different threshold values to find the best balance between
label quality and label coverage. Check the summary output to decide.

Usage:
    python apply_threshold.py

To experiment with thresholds, change THRESHOLD at the top.
Suggested values to try: 0.3, 0.4, 0.5, 0.6, 0.7

Input:  labeled_main_dataset.json
Output: labeled_main_dataset_t{threshold}.json
        (e.g. labeled_main_dataset_t0.5.json)
"""

import json
import numpy as np
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — change THRESHOLD to experiment
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLD  = 0.8

INPUT_FILE  = "../data/JSON/labeled_main_dataset.json"
OUTPUT_FILE = f"../data/JSON/labeled_main_dataset_t{THRESHOLD}.json"

# ── Value list ────────────────────────────────────────────────────────────────

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal',
    'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature',
    'Universalism: tolerance', 'Universalism: objectivity'
]

# ── Helper ────────────────────────────────────────────────────────────────────

def apply_threshold(text_obj: dict, threshold: float) -> dict:
    """
    Filters predicted_values to only keep labels where the predicted
    probability is >= threshold. Updates predicted_binary accordingly.
    Returns a new dict — does not modify the original.
    """
    probs  = text_obj.get("predicted_probabilities", [])
    result = dict(text_obj)

    if not probs or len(probs) != len(FEATURES):
        return result

    new_binary = [1 if p >= threshold else 0 for p in probs]
    new_values = [FEATURES[i] for i, p in enumerate(probs) if p >= threshold]

    result["predicted_binary"] = new_binary
    result["predicted_values"] = new_values
    return result


def summarise(data: list, label: str):
    """Print a compact summary of label coverage for a dataset."""
    all_texts = (
        [item["op_context"] for item in data] +
        [item["winning_rebuttal"] for item in data] +
        [item["losing_rebuttal"] for item in data]
    )

    label_counts  = [len(t.get("predicted_values", [])) for t in all_texts]
    zero_labels   = sum(1 for c in label_counts if c == 0)
    value_counter = Counter(v for t in all_texts for v in t.get("predicted_values", []))
    total_texts   = len(all_texts)

    print(f"\n  {label}")
    print(f"  {'─' * 55}")
    print(f"  Total texts:          {total_texts}")
    print(f"  Mean labels/text:     {np.mean(label_counts):.2f}")
    print(f"  Texts with 0 labels:  {zero_labels}  ({zero_labels/total_texts*100:.1f}%)")
    print(f"  Unique values used:   {len(value_counter)}")
    print()
    print(f"  {'Value':<35}  {'Count':>6}  {'% texts':>8}")
    print(f"  {'─' * 52}")
    for val, count in sorted(value_counter.items(), key=lambda x: -x[1]):
        print(f"  {val:<35}  {count:>6}  {count/total_texts*100:>7.1f}%")

    # Values completely dropped
    dropped = [v for v in FEATURES if v not in value_counter]
    if dropped:
        print(f"\n  Values completely dropped: {len(dropped)}")
        for v in dropped:
            print(f"    - {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Applying threshold: {THRESHOLD}")
    print(f"Total pairs: {len(data)}")

    # ── Show before/after comparison ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Comparison: before vs after threshold {THRESHOLD}")
    print("=" * 65)

    summarise(data, f"BEFORE (no threshold)")

    # Apply threshold
    cleaned = []
    for item in data:
        cleaned.append({
            **item,
            "op_context":       apply_threshold(item["op_context"],       THRESHOLD),
            "winning_rebuttal": apply_threshold(item["winning_rebuttal"], THRESHOLD),
            "losing_rebuttal":  apply_threshold(item["losing_rebuttal"],  THRESHOLD),
        })

    summarise(cleaned, f"AFTER  (threshold = {THRESHOLD})")

    # ── Value overlap comparison ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Value overlap: OP vs winning/losing  (Jaccard similarity)")
    print("=" * 65)

    for label, dataset in [("Before", data), ("After", cleaned)]:
        win_overlaps  = []
        lose_overlaps = []
        for item in dataset:
            op_vals   = set(item["op_context"].get("predicted_values", []))
            win_vals  = set(item["winning_rebuttal"].get("predicted_values", []))
            lose_vals = set(item["losing_rebuttal"].get("predicted_values", []))

            win_j  = len(op_vals & win_vals)  / max(len(op_vals | win_vals),  1)
            lose_j = len(op_vals & lose_vals) / max(len(op_vals | lose_vals), 1)
            win_overlaps.append(win_j)
            lose_overlaps.append(lose_j)

        print(f"\n  {label} (threshold = {THRESHOLD if label == 'After' else 'none'}):")
        print(f"  {'':25}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
        print(f"  {'─' * 52}")
        print(f"  {'Winning (delta)':25}  {np.mean(win_overlaps):>8.3f}  {np.median(win_overlaps):>8.3f}  {np.std(win_overlaps):>8.3f}")
        print(f"  {'Losing (no delta)':25}  {np.mean(lose_overlaps):>8.3f}  {np.median(lose_overlaps):>8.3f}  {np.std(lose_overlaps):>8.3f}")

    # ── Save cleaned file ─────────────────────────────────────────────────────
    print(f"\nSaving cleaned dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Done. Change THRESHOLD and re-run to experiment with other values.")


if __name__ == "__main__":
    main()