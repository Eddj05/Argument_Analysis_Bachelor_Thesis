"""
Step 9a — Prepare Prompts (Three Conditions)

Reads the labeled dataset with strategy (output of step 6) and generates
three prompt files — one per experimental condition:

  - text_only:        raw argument text, no value labels
  - text_and_labels:  raw argument text + value labels
  - labels_only:      value labels only, no raw text

The value strategy label is intentionally withheld from all prompts.
It is only used during analysis (step 12) to split results.

Usage:
    python step9a_prepare_prompts_v2.py

Input:  labeled_data_with_strategy.csv
Output: prompts_text_only.json
        prompts_text_and_labels.json
        prompts_labels_only.json
"""

import ast
import json
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE = "../data/CSV/labeled_data_with_strategy.csv"

OUTPUT_FILES = {
    "text_only":       "../data/JSON/prompts_text_only.json",
    "text_and_labels": "../data/JSON/prompts_text_and_labels.json",
    "labels_only":     "../data/JSON/prompts_labels_only.json",
}

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_text_only(op_text: str, reply_text: str) -> str:
    return f"""You are analysing an online debate from Reddit's ChangeMyView.
An original poster can reward a reply with a delta if the reply persuades 
them to change their view. Deltas are rare. Most replies do NOT receive one. 
Only replies that directly address the OP's core reasoning and offer a 
genuinely new perspective tend to succeed.

Original poster:
"{op_text}"

Reply:
"{reply_text}"

Think step by step:
1. What is the original poster's core argument?
2. Does the reply directly challenge that core argument, or does it talk past it?
3. Is there a genuinely new perspective here that could shift the OP's view, 
   or is this just a counterargument they likely already considered?
4. Given that most replies fail to persuade, does this one clear that bar?

Based on your reasoning, will this reply receive a delta?
Answer with only: yes or no"""


def build_text_and_labels(op_text: str, op_values: list,
                           reply_text: str, reply_values: list) -> str:
    return f"""You are analysing an online debate from Reddit's ChangeMyView.
An original poster can reward a reply with a delta if the reply persuades
them to change their view. Deltas are rare. Most replies do NOT receive one.
Only replies that directly address the OP's core reasoning and offer a
genuinely new perspective tend to succeed.

Original poster:
"{op_text}"
OP values: {", ".join(op_values)}

Reply:
"{reply_text}"
Reply values: {", ".join(reply_values)}

Think step by step:
1. What is the original poster's core argument, and which values drive it?
2. Does the reply directly challenge that core argument, or does it talk past it?
3. Do the reply's values align with, reframe, or conflict with the OP's values?
4. Does the reply offer a genuinely new perspective that resonates with the OP's value frame?
5. Given that most replies fail to persuade, does this one clear that bar?

Based on your reasoning, will this reply receive a delta?
Answer with only: yes or no"""


def build_labels_only(op_values: list, reply_values: list) -> str:
    return f"""You are analysing an online debate from Reddit's ChangeMyView.
An original poster can reward a reply with a delta if the reply persuades
them to change their view.

We hid the text of the debate. You do NOT have access to the actual text. You will only see 
the human values each author appeals to, based on Schwartz's theory of basic 
human values. Make your best guess based on these value labels alone. 
Delta's are rare. Usually, only the best comments will receive a delta.
Would an argument that uses these values receive a delta from the original poster?


OP values: {", ".join(op_values)}
Reply values: {", ".join(reply_values)}

Think step by step:
1. What does the OP's value profile suggest about what they fundamentally care about?
2. Do the reply's values align with, reframe, or conflict with the OP's values?
3. Is the value relationship between OP and reply likely to feel relevant and credible to the OP?
4. Based on this value relationship, how likely is it that the reply resonates with the OP?

Based on your reasoning, will this reply receive a delta?
Answer with only: yes or no"""


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

    all_prompts = {condition: [] for condition in OUTPUT_FILES}

    for _, row in df.iterrows():
        meta = {
            "pair_id":    str(row["pair_id"]),
            "strategy":   str(row["strategy"]),
            "delta_true": int(row["delta"]),
        }

        all_prompts["text_only"].append({
            **meta,
            "condition": "text_only",
            "prompt": build_text_only(
                op_text    = str(row["op_text"]),
                reply_text = str(row["reply_text"]),
            )
        })

        all_prompts["text_and_labels"].append({
            **meta,
            "condition": "text_and_labels",
            "prompt": build_text_and_labels(
                op_text      = str(row["op_text"]),
                op_values    = row["op_values"],
                reply_text   = str(row["reply_text"]),
                reply_values = row["reply_values"],
            )
        })

        all_prompts["labels_only"].append({
            **meta,
            "condition": "labels_only",
            "prompt": build_labels_only(
                op_values    = row["op_values"],
                reply_values = row["reply_values"],
            )
        })

    print()
    for condition, filepath in OUTPUT_FILES.items():
        prompts = all_prompts[condition]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(prompts)} prompts → {filepath}")


if __name__ == "__main__":
    main()