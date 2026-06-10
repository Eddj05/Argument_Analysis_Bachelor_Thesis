"""
dataset_analysis.py — Analyze the labeled dataset

Reads labeled_main_dataset.json and prints a detailed summary of:
  1. Dataset overview (total pairs, texts, delta distribution)
  2. Value label distribution (how often each value is assigned)
  3. Confidence scores per value (mean predicted probability)
  4. Labels per text (how many values assigned on average)
  5. Low confidence predictions (potential noise in the data)

Usage:
    python dataset_analysis.py

Input:  labeled_main_dataset.json
"""

import json
import numpy as np
from collections import Counter, defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE = "../data/JSON/labeled_main_dataset_t0.5.json"

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal',
    'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature',
    'Universalism: tolerance', 'Universalism: objectivity'
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)

def subsection(title):
    print(f"\n  ── {title} ──")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Collect all texts ─────────────────────────────────────────────────────
    op_texts      = []
    winning_texts = []
    losing_texts  = []

    for item in data:
        op_texts.append(item["op_context"])
        winning_texts.append(item["winning_rebuttal"])
        losing_texts.append(item["losing_rebuttal"])

    all_texts = op_texts + winning_texts + losing_texts
    text_types = (
        [("op", t) for t in op_texts] +
        [("winning", t) for t in winning_texts] +
        [("losing", t) for t in losing_texts]
    )

    # ── 1. Dataset Overview ───────────────────────────────────────────────────
    section("1. Dataset Overview")

    print(f"\n  Total pairs:          {len(data)}")
    print(f"  Total texts:          {len(all_texts)}")
    print(f"    OP posts:           {len(op_texts)}")
    print(f"    Winning rebuttals:  {len(winning_texts)}")
    print(f"    Losing rebuttals:   {len(losing_texts)}")

    # Delta distribution — winning = delta, losing = no delta
    print(f"\n  Delta distribution:")
    print(f"    Delta (winning):    {len(winning_texts)}  ({len(winning_texts)/len(all_texts)*100:.1f}% of all texts)")
    print(f"    No delta (losing):  {len(losing_texts)}  ({len(losing_texts)/len(all_texts)*100:.1f}% of all texts)")
    print(f"    OP (no label):      {len(op_texts)}  ({len(op_texts)/len(all_texts)*100:.1f}% of all texts)")

    # ── 2. Value Label Distribution ───────────────────────────────────────────
    section("2. Value Label Distribution")

    # Count how often each value is assigned across all texts
    value_counts = Counter()
    for _, text in text_types:
        for val in text.get("predicted_values", []):
            value_counts[val] += 1

    total_texts = len(all_texts)
    print(f"\n  {'Value':<35}  {'Count':>6}  {'% of texts':>10}  {'Rank':>5}")
    print("  " + "-" * 60)
    for rank, (val, count) in enumerate(sorted(value_counts.items(), key=lambda x: -x[1]), 1):
        pct = count / total_texts * 100
        print(f"  {val:<35}  {count:>6}  {pct:>9.1f}%  {rank:>5}")

    # Values never assigned
    never_assigned = [v for v in FEATURES if v not in value_counts]
    if never_assigned:
        print(f"\n  Values never assigned ({len(never_assigned)}):")
        for v in never_assigned:
            print(f"    - {v}")

    # Split by text type
    for label, texts in [("OP posts", op_texts), ("Winning rebuttals", winning_texts), ("Losing rebuttals", losing_texts)]:
        subsection(f"Value distribution — {label}")
        counts = Counter()
        for t in texts:
            for v in t.get("predicted_values", []):
                counts[v] += 1
        print(f"  {'Value':<35}  {'Count':>6}  {'%':>8}")
        print("  " + "-" * 52)
        for val, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {val:<35}  {count:>6}  {count/len(texts)*100:>7.1f}%")

    # ── 3. Confidence Scores per Value ────────────────────────────────────────
    section("3. Confidence Scores per Value (Mean Predicted Probability)")

    # Collect all probabilities per value
    probs_per_value = defaultdict(list)
    for _, text in text_types:
        probs = text.get("predicted_probabilities", [])
        if len(probs) == len(FEATURES):
            for i, val in enumerate(FEATURES):
                probs_per_value[val].append(probs[i])

    print(f"\n  {'Value':<35}  {'Mean prob':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 72)
    for val in sorted(probs_per_value.keys(), key=lambda v: -np.mean(probs_per_value[v])):
        ps = np.array(probs_per_value[val])
        print(f"  {val:<35}  {ps.mean():>10.4f}  {ps.std():>8.4f}  {ps.min():>8.4f}  {ps.max():>8.4f}")

    # ── 4. Labels per Text ────────────────────────────────────────────────────
    section("4. Number of Labels Assigned per Text")

    label_counts = [len(t.get("predicted_values", [])) for _, t in text_types]
    label_counter = Counter(label_counts)

    print(f"\n  Mean labels per text:    {np.mean(label_counts):.2f}")
    print(f"  Median labels per text:  {np.median(label_counts):.1f}")
    print(f"  Min labels:              {min(label_counts)}")
    print(f"  Max labels:              {max(label_counts)}")

    print(f"\n  Distribution:")
    print(f"  {'# labels':>10}  {'Count':>8}  {'%':>8}")
    print("  " + "-" * 30)
    for n_labels, count in sorted(label_counter.items()):
        print(f"  {n_labels:>10}  {count:>8}  {count/total_texts*100:>7.1f}%")

    # Texts with zero labels
    zero_label = sum(1 for c in label_counts if c == 0)
    if zero_label > 0:
        print(f"\n  Warning: {zero_label} texts received no value labels ({zero_label/total_texts*100:.1f}%)")

    # ── 5. Low Confidence Predictions ────────────────────────────────────────
    section("5. Low Confidence Predictions (threshold: max prob < 0.5)")

    low_conf_texts = []
    for text_type, text in text_types:
        probs = text.get("predicted_probabilities", [])
        if probs:
            max_prob = max(probs)
            if max_prob < 0.5:
                low_conf_texts.append({
                    "type":      text_type,
                    "max_prob":  max_prob,
                    "values":    text.get("predicted_values", []),
                })

    print(f"\n  Total low-confidence texts: {len(low_conf_texts)} / {total_texts} ({len(low_conf_texts)/total_texts*100:.1f}%)")

    if low_conf_texts:
        print(f"\n  {'Type':<12}  {'Max prob':>10}  {'Assigned values'}")
        print("  " + "-" * 60)
        for item in sorted(low_conf_texts, key=lambda x: x["max_prob"])[:20]:
            vals = ", ".join(item["values"]) if item["values"] else "none"
            print(f"  {item['type']:<12}  {item['max_prob']:>10.4f}  {vals}")
        if len(low_conf_texts) > 20:
            print(f"  ... and {len(low_conf_texts) - 20} more")

    # ── 6. Winning vs Losing value overlap ───────────────────────────────────
    section("6. Value Overlap Between Winning and Losing Rebuttals")

    overlaps = []
    for item in data:
        win_vals  = set(item["winning_rebuttal"].get("predicted_values", []))
        lose_vals = set(item["losing_rebuttal"].get("predicted_values", []))
        op_vals   = set(item["op_context"].get("predicted_values", []))

        win_op_overlap  = len(win_vals  & op_vals) / max(len(win_vals  | op_vals), 1)
        lose_op_overlap = len(lose_vals & op_vals) / max(len(lose_vals | op_vals), 1)
        overlaps.append((win_op_overlap, lose_op_overlap))

    win_overlaps  = [o[0] for o in overlaps]
    lose_overlaps = [o[1] for o in overlaps]

    print(f"\n  Jaccard similarity between OP and rebuttal value sets:")
    print(f"  {'':25}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
    print("  " + "-" * 52)
    print(f"  {'Winning (delta)':25}  {np.mean(win_overlaps):>8.3f}  {np.median(win_overlaps):>8.3f}  {np.std(win_overlaps):>8.3f}")
    print(f"  {'Losing (no delta)':25}  {np.mean(lose_overlaps):>8.3f}  {np.median(lose_overlaps):>8.3f}  {np.std(lose_overlaps):>8.3f}")

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()