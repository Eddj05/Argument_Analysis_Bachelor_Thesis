"""
Verify ollama and model Setup

Checks that:
  1. Ollama is running and reachable
  2. Both llama3 and mistral are installed
  3. Both models can actually respond to a simple prompt

Run this before step 9 to make sure everything is working.

Usage:
    python step8_verify_setup.py
"""

import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODELS     = ["llama3", "mistral"]
TEST_PROMPT = "Reply with only the word: working"

# ── Checks ────────────────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Check if Ollama is reachable at all."""
    try:
        response = requests.get(f"{OLLAMA_URL}", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_model_installed(model: str) -> bool:
    """Check if a model is listed in Ollama's local models."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        models = [m["name"] for m in response.json().get("models", [])]
        # Match loosely — "llama3" matches "llama3:latest" etc.
        return any(model in m for m in models)
    except Exception:
        return False


def check_model_responds(model: str) -> tuple:
    """
    Send a simple test prompt to a model and check it responds.
    Returns (success: bool, response_text: str)
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": TEST_PROMPT,
                "stream": False,
            },
            timeout=60,
        )
        if response.status_code == 200:
            text = response.json().get("response", "").strip()
            return True, text
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout — model took too long to respond"
    except Exception as e:
        return False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Step 8 — Ollama Setup Verification")
    print("=" * 50)
    print()

    all_good = True

    # Check 1: Ollama running
    print("[ 1/3 ] Checking Ollama is running...")
    if check_ollama_running():
        print("        ✓ Ollama is running at", OLLAMA_URL)
    else:
        print("        ✗ Ollama is NOT running.")
        print("          Fix: open a terminal and run: ollama serve")
        print("          Then re-run this script.")
        return
    print()

    # Check 2: models installed
    print("[ 2/3 ] Checking models are installed...")
    for model in MODELS:
        if check_model_installed(model):
            print(f"        ✓ {model} is installed")
        else:
            print(f"        ✗ {model} is NOT installed")
            print(f"          Fix: run: ollama pull {model}")
            all_good = False
    print()

    # Check 3: models respond
    print("[ 3/3 ] Sending test prompt to each model...")
    print(f"        Prompt: \"{TEST_PROMPT}\"")
    print()
    for model in MODELS:
        print(f"        Testing {model}...", end=" ", flush=True)
        success, text = check_model_responds(model)
        if success:
            print(f"✓ responded: \"{text}\"")
        else:
            print(f"✗ failed: {text}")
            all_good = False
    print()

    # Summary
    print("=" * 50)
    if all_good:
        print("All checks passed")
    else:
        print("Some checks failed")
    print("=" * 50)


if __name__ == "__main__":
    main()