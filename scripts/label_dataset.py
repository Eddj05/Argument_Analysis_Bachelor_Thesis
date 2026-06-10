import json
import os
import math
import numpy as np
import torch
from setfit import SetFitModel

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "../data/JSON/triage_pairs.json"
OUTPUT_FILE = "../data/JSON/labeled_main_dataset.json"
MODEL_PATH = "../classifier/setfit-modernBERT-base-full-dataset"

# Memory Optimization Parameters for GPU (MPS)
BATCH_SIZE = 128           # Number of records read from JSON at once
GPU_MICRO_BATCH = 16       # Keeps VRAM allocation safely within Apple GPU limits
MAX_SEQUENCE_LENGTH = 512  # Preserves 80.3% of your corpus intact

FEATURES = [
    'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
    'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
    'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
    'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
    'Universalism: objectivity'
]

# ==========================================
# JSON SERIALIZATION HELPER
# ==========================================
def to_standard_list(obj):
    """
    Recursively converts PyTorch Tensors, NumPy arrays, and NumPy scalars 
    into standard Python lists/types to ensure JSON serializability.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [to_standard_list(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64, float)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    return obj

# ==========================================
# 1. LOAD DATA & MODEL
# ==========================================
print(f"Loading main dataset from: {INPUT_FILE}")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loading fine-tuned SetFit model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model path: {MODEL_PATH}")

model = SetFitModel.from_pretrained(MODEL_PATH)

# Force GPU (MPS) execution
print("Moving model to Apple Silicon GPU (MPS)...")
model.to("mps")

# Enforce the 512-token limit
print(f"Enforcing maximum sequence length of {MAX_SEQUENCE_LENGTH} tokens...")
model.model_body.max_seq_length = MAX_SEQUENCE_LENGTH

# ==========================================
# 2. BATCH INFERENCE PROCESS
# ==========================================
total_records = len(data)
total_batches = math.ceil(total_records / BATCH_SIZE)
print(f"Starting optimized GPU inference on {total_records} records over {total_batches} batches...")

def get_predicted_labels(binary_vector):
    return [FEATURES[idx] for idx, val in enumerate(binary_vector) if val == 1.0 or val == 1]

# Disable gradient overhead for faster processing
with torch.inference_mode():
    for i in range(0, total_records, BATCH_SIZE):
        batch_idx = (i // BATCH_SIZE) + 1
        batch = data[i:i + BATCH_SIZE]
        print(f"  -> Processing batch {batch_idx}/{total_batches}...")
        
        # Extract texts for this batch
        op_texts = [item["op_context"]["text"] for item in batch]
        win_texts = [item["winning_rebuttal"]["text"] for item in batch]
        lose_texts = [item["losing_rebuttal"]["text"] for item in batch]
        
        batch_texts = op_texts + win_texts + lose_texts
        
        # Predict binary labels
        batch_preds = model.predict(batch_texts, batch_size=GPU_MICRO_BATCH)
        
        # Predict probabilities
        try:
            batch_probs = model.predict_proba(batch_texts, batch_size=GPU_MICRO_BATCH)
        except AttributeError:
            batch_probs = batch_preds
            
        # Clean PyTorch/NumPy types into standard Python lists/scalars
        batch_preds = to_standard_list(batch_preds)
        batch_probs = to_standard_list(batch_probs)
        
        # Slice the results back to individual arrays
        m = len(batch)
        op_preds_b, win_preds_b, lose_preds_b = batch_preds[:m], batch_preds[m:2*m], batch_preds[2*m:]
        op_probs_b, win_probs_b, lose_probs_b = batch_probs[:m], batch_probs[m:2*m], batch_probs[2*m:]
        
        # Save the predictions back into the batch items
        for idx, item in enumerate(batch):
            # OP predictions
            item["op_context"]["predicted_binary"] = op_preds_b[idx]
            item["op_context"]["predicted_probabilities"] = op_probs_b[idx]
            item["op_context"]["predicted_values"] = get_predicted_labels(op_preds_b[idx])

            # Winning Rebuttal predictions
            item["winning_rebuttal"]["predicted_binary"] = win_preds_b[idx]
            item["winning_rebuttal"]["predicted_probabilities"] = win_probs_b[idx]
            item["winning_rebuttal"]["predicted_values"] = get_predicted_labels(win_preds_b[idx])

            # Losing Rebuttal predictions
            item["losing_rebuttal"]["predicted_binary"] = lose_preds_b[idx]
            item["losing_rebuttal"]["predicted_probabilities"] = lose_probs_b[idx]
            item["losing_rebuttal"]["predicted_values"] = get_predicted_labels(lose_preds_b[idx])

# ==========================================
# 3. SAVE OUTPUT
# ==========================================
print(f"Saving updated dataset to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Inference completed successfully.")