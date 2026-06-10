"""
evaluation.py — Baseline + LLM Evaluation + Strategy Analysis

Prints clean comparative tables instead of individual reports.

Usage:
    python evaluation.py

Input:  results_combined.csv
        labeled_data_with_strategy.csv
"""

import ast
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_FILE = "../data/CSV/results_combined.csv"
DATA_FILE    = "../data/CSV/labeled_data_with_strategy.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_values(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def jaccard_baseline(op_values, reply_values):
    return "yes" if set(op_values) & set(reply_values) else "no"


def read_csv_safe(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.read_csv(filepath, on_bad_lines="skip", engine="python")


def metrics(y_true, y_pred):
    """Compute macro F1, accuracy, yes-precision, yes-recall for a set of predictions."""
    # Filter out unclear
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p in ("yes", "no")]
    if not pairs:
        return None
    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]

    if len(set(yp)) == 1:
        # Degenerate — only one class predicted
        acc = accuracy_score(yt, yp)
        return {
            "n":        len(yt),
            "acc":      round(acc, 3),
            "macro_f1": "—",
            "prec_yes": "—",
            "rec_yes":  "—",
            "note":     f"all '{yp[0]}'"
        }

    return {
        "n":        len(yt),
        "acc":      round(accuracy_score(yt, yp), 3),
        "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
        "prec_yes": round(precision_score(yt, yp, pos_label="yes", zero_division=0), 3),
        "rec_yes":  round(recall_score(yt, yp, pos_label="yes", zero_division=0), 3),
        "note":     ""
    }


def print_section(title):
    print()
    print("=" * 75)
    print(f"  {title}")
    print("=" * 75)


def print_comparison_table(rows, title=None):
    """
    rows = list of dicts with keys:
      label, n, acc, macro_f1, prec_yes, rec_yes, note
    """
    if title:
        print(f"\n  {title}")
    header = f"  {'Model / Condition':<42}  {'n':>6}  {'acc':>6}  {'F1':>6}  {'prec':>6}  {'rec':>6}"
    print(header)
    print("  " + "-" * 72)
    for row in rows:
        if row is None:
            print("  " + "-" * 72)
            continue
        note = f"  ← {row['note']}" if row.get("note") else ""
        print(
            f"  {row['label']:<42}"
            f"  {row['n']:>6}"
            f"  {str(row['acc']):>6}"
            f"  {str(row['macro_f1']):>6}"
            f"  {str(row['prec_yes']):>6}"
            f"  {str(row['rec_yes']):>6}"
            f"{note}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Reading {DATA_FILE}...")
    data = read_csv_safe(DATA_FILE)
    data["op_values"]    = data["op_values"].apply(parse_values)
    data["reply_values"] = data["reply_values"].apply(parse_values)
    data["pair_id"]      = data["pair_id"].astype(str)
    data["delta_label"]  = data["delta"].apply(lambda x: "yes" if x == 1 else "no")
    data["baseline_pred"] = data.apply(
        lambda r: jaccard_baseline(r["op_values"], r["reply_values"]), axis=1
    )

    print(f"Reading {RESULTS_FILE}...")
    results = read_csv_safe(RESULTS_FILE)
    results["pair_id"]     = results["pair_id"].astype(str)
    results["delta_label"] = results["delta_true"].apply(lambda x: "yes" if x == 1 else "no")
    results["prediction"]  = results["prediction"].astype(str).str.lower().str.strip()

    models     = sorted(results["model"].unique())
    conditions = sorted(results["condition"].unique())
    strategies = sorted(results["strategy"].dropna().unique())

    print(f"\n  Models:     {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Strategies: {strategies}")

    # ── 1. Overall comparison — by condition ──────────────────────────────────
    print_section("1. Overall Evaluation by Condition  (RQ2)")
    print("  Columns: accuracy  |  macro F1  |  precision (yes)  |  recall (yes)")

    # Baseline row
    bm = metrics(data["delta_label"].tolist(), data["baseline_pred"].tolist())
    bm["label"] = "Baseline (Jaccard overlap)"

    for condition in conditions:
        rows = [None, bm]  # separator + baseline
        for model in models:
            group = results[
                (results["model"] == model) &
                (results["condition"] == condition)
            ]
            if len(group) == 0:
                continue
            m = metrics(group["delta_label"].tolist(), group["prediction"].tolist())
            if m:
                m["label"] = f"{model}"
                rows.append(m)
        print_comparison_table(rows, title=f"Condition: {condition}")

    # ── 2. Overall comparison — by model ─────────────────────────────────────
    print_section("2. Condition Comparison per Model  (RQ2)")
    print("  Does adding value labels help? text_only vs text_and_labels vs labels_only")

    for model in models:
        rows = [None, bm]
        for condition in conditions:
            group = results[
                (results["model"] == model) &
                (results["condition"] == condition)
            ]
            if len(group) == 0:
                continue
            m = metrics(group["delta_label"].tolist(), group["prediction"].tolist())
            if m:
                m["label"] = f"{condition}"
                rows.append(m)
        print_comparison_table(rows, title=f"Model: {model}")

    # ── 3. Strategy split ─────────────────────────────────────────────────────
    print_section("3. Evaluation by Value Strategy  (RQ3)")
    print("  Does prediction accuracy differ across aligned / reframing / opposition?")

    for strategy in strategies:
        sub = data[data["strategy"] == strategy]
        bm_s = metrics(sub["delta_label"].tolist(), sub["baseline_pred"].tolist())
        if bm_s:
            bm_s["label"] = "Baseline (Jaccard overlap)"

        rows = [None, bm_s]
        for model in models:
            for condition in conditions:
                group = results[
                    (results["model"] == model) &
                    (results["condition"] == condition) &
                    (results["strategy"] == strategy)
                ]
                if len(group) == 0:
                    continue
                m = metrics(group["delta_label"].tolist(), group["prediction"].tolist())
                if m:
                    m["label"] = f"{model}  [{condition}]"
                    rows.append(m)

        print_comparison_table(rows, title=f"Strategy: {strategy.upper()}")

    # ── 4. F1 summary heatmap ─────────────────────────────────────────────────
    print_section("4. Macro F1 Summary Table")
    print("  Quick overview — macro F1 for every model × condition × strategy\n")

    col_headers = [f"{m[:8]}+{c[:4]}" for m in models for c in conditions]
    print(f"  {'':15}", end="")
    for m in models:
        for c in conditions:
            label = f"{m.split(':')[0]}:{c[:4]}"
            print(f"  {label:>14}", end="")
    print()
    print("  " + "-" * (15 + 16 * len(models) * len(conditions)))

    all_strats = ["overall"] + list(strategies)
    for strategy in all_strats:
        print(f"  {strategy:<15}", end="")
        for model in models:
            for condition in conditions:
                if strategy == "overall":
                    group = results[
                        (results["model"] == model) &
                        (results["condition"] == condition)
                    ]
                else:
                    group = results[
                        (results["model"] == model) &
                        (results["condition"] == condition) &
                        (results["strategy"] == strategy)
                    ]
                if len(group) == 0:
                    print(f"  {'—':>14}", end="")
                    continue
                m = metrics(group["delta_label"].tolist(), group["prediction"].tolist())
                val = str(m["macro_f1"]) if m else "—"
                print(f"  {val:>14}", end="")
        print()

    # ── 5. Delta rates ────────────────────────────────────────────────────────
    print_section("5. Delta Rates per Value Strategy")
    print(f"\n  {'Strategy':<15}  {'n pairs':>8}  {'n delta':>8}  {'delta rate':>10}  {'% of dataset':>12}")
    print("  " + "-" * 58)

    delta_data = results[
        (results["condition"] == conditions[0]) &
        (results["model"] == models[0])
    ][["pair_id", "strategy", "delta_true"]].drop_duplicates()

    total = len(delta_data)
    for strategy in sorted(
        strategies,
        key=lambda s: -delta_data[delta_data["strategy"] == s]["delta_true"].mean()
    ):
        sub        = delta_data[delta_data["strategy"] == strategy]
        n          = len(sub)
        n_delta    = int(sub["delta_true"].sum())
        delta_rate = sub["delta_true"].mean()
        pct        = n / total * 100
        print(f"  {strategy:<15}  {n:>8}  {n_delta:>8}  {delta_rate:>9.1%}  {pct:>11.1f}%")

    print()


if __name__ == "__main__":
    main()