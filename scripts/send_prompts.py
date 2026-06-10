"""
Run LLM Predictions on Prepared Prompts

Reads the three prompt files (output of prepare_prompts.py) and sends
every prompt to each model. Saves all predictions to one results.csv
with a 'condition' column so you can compare conditions in analysis.

Works identically locally and on Habrok — just change the config block.

Local (Ollama):
  - Set MODE = "ollama"
  - Set OLLAMA_MODELS to your local model names (check: ollama list)

Habrok (HuggingFace):
  - Set MODE = "huggingface"
  - HF_MODELS is used automatically

Usage:
    python inference.py

Input:  prompts_text_only.json
        prompts_text_and_labels.json
        prompts_labels_only.json
Output: results.csv
"""

import json
import csv
import time
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — only change this block when switching between local and Habrok
# ══════════════════════════════════════════════════════════════════════════════

# "ollama" for local testing, "huggingface" for Habrok
MODE = "ollama"

# Local Ollama model names — check yours with: ollama list
OLLAMA_MODELS = [
    "llama3.2:3b",       # small test model, fits in ~2GB RAM
    # "llama3:latest",   # uncomment for full model
    # "mistral:latest",  # uncomment for mistral
]

# Habrok HuggingFace model names, only used when MODE = "huggingface"
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# Input prompt files — must match OUTPUT_FILES in prepare_prompts.py
PROMPT_FILES = [
    "../data/JSON/prompts_text_only.json",
    "../data/JSON/prompts_text_and_labels.json",
    "../data/JSON/prompts_labels_only.json",
]

OUTPUT_FILE = "../data/CSV/results.csv"
OLLAMA_URL  = "http://localhost:11434"

# Limit prompts per file for quick local tests — set None to run all
LIMIT = 4

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

def predict_ollama(prompt: str, model: str, temperature: float = 0) -> str:
    import requests
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": 42,
                }
            },
            timeout=300,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return f"ERROR: HTTP {response.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


def load_hf_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


def predict_huggingface(prompt: str, tokenizer, model,
                        do_sample: bool = False, temperature: float = 1.0) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# PARSE YES / NO
# ══════════════════════════════════════════════════════════════════════════════

def parse_prediction(raw: str) -> str:
    cleaned = raw.lower().strip()
    if cleaned.startswith("yes"): return "yes"
    if cleaned.startswith("no"):  return "no"
    if "yes" in cleaned:          return "yes"
    if "no" in cleaned:           return "no"
    return "unclear"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_ollama(model: str, all_prompts: dict) -> list:
    results = []
    for prompt_file, prompts in all_prompts.items():
        print(f"  [{model}] {prompt_file} ({len(prompts)} prompts)...")
        for i, item in enumerate(prompts, 1):
            print(f"    [{i}/{len(prompts)}] pair {item['pair_id']} ({item['condition']})...", end=" ", flush=True)
            

            temp = 0.8 if item["condition"] == "labels_only" else 0
            
            start = time.time()
            raw   = predict_ollama(item["prompt"], model, temperature=temp)
            pred  = parse_prediction(raw)
            print(f"{pred}  ({time.time()-start:.1f}s)")
            results.append({
                "pair_id":      item["pair_id"],
                "strategy":     item["strategy"],
                "delta_true":   item["delta_true"],
                "condition":    item["condition"],
                "model":        model,
                "raw_response": raw,
                "prediction":   pred,
            })
    return results


def run_hf(model_name: str, all_prompts: dict) -> list:
    tokenizer, model = load_hf_model(model_name)
    results = []
    for prompt_file, prompts in all_prompts.items():
        print(f"  [{model_name}] {prompt_file} ({len(prompts)} prompts)...")
        for i, item in enumerate(prompts, 1):
            print(f"    [{i}/{len(prompts)}] pair {item['pair_id']} ({item['condition']})...", end=" ", flush=True)
            do_sample   = item["condition"] == "labels_only"
            temperature = 0.5 if do_sample else 1.0

            start = time.time()
            raw   = predict_huggingface(item["prompt"], tokenizer, model,
                                        do_sample=do_sample, temperature=temperature)
            pred  = parse_prediction(raw)
            print(f"{pred}  ({time.time()-start:.1f}s)")
            results.append({
                "pair_id":      item["pair_id"],
                "strategy":     item["strategy"],
                "delta_true":   item["delta_true"],
                "condition":    item["condition"],
                "model":        model_name,
                "raw_response": raw,
                "prediction":   pred,
            })
    del model, tokenizer
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def main():
    # Load all prompt files
    print("Loading prompt files...")
    all_prompts = {}
    for prompt_file in PROMPT_FILES:
        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)
        if LIMIT is not None:
            prompts = prompts[:LIMIT]
        all_prompts[prompt_file] = prompts
        print(f"  {prompt_file}: {len(prompts)} prompts")

    models = OLLAMA_MODELS if MODE == "ollama" else HF_MODELS
    total  = sum(len(p) for p in all_prompts.values())

    print()
    print(f"Mode:          {MODE}")
    print(f"Models:        {models}")
    print(f"Total prompts: {total} per model")
    print()

    # Run inference
    all_results = []
    if MODE == "ollama":
        for model in models:
            print(f"Running {model}...")
            all_results.extend(run_ollama(model, all_prompts))
            print()
    elif MODE == "huggingface":
        for model_name in models:
            print(f"Running {model_name}...")
            all_results.extend(run_hf(model_name, all_prompts))
            print()

    # Save
    fieldnames = ["pair_id", "strategy", "delta_true", "condition",
                  "model", "raw_response", "prediction"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved {len(all_results)} predictions to {OUTPUT_FILE}")
    print()

    # Summary
    df = pd.DataFrame(all_results)
    df["correct"] = df.apply(
        lambda r: (r["prediction"] == "yes") == (r["delta_true"] == 1), axis=1
    )
    print("Quick accuracy summary:")
    print(f"  {'Model':<40} {'Condition':<22} {'Accuracy':>8}  {'Unclear':>7}")
    print("  " + "-" * 80)
    for (model, condition), group in df.groupby(["model", "condition"]):
        acc     = group["correct"].mean()
        unclear = (group["prediction"] == "unclear").sum()
        print(f"  {model:<40} {condition:<22} {acc:>7.0%}  {unclear:>7}")
    print()


if __name__ == "__main__":
    main()