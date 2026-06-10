from datasets import load_dataset
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

# 1. Define your features
features = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
            'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
            'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
            'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
            'Universalism: objectivity']


# 2. Recreate the exact test dataset
def encode_labels(record):
    encoded = [0.0] * len(features)
    for val in record["values"]:
        if val in features:
            encoded[features.index(val)] = 1.0
    return {"label": encoded}


print("Loading and preparing dataset...")
dataset = load_dataset("json", data_files="complete_dataset_test.json", split="train")
dataset = dataset.map(encode_labels)

# We use the exact same seed (3407) to ensure we get the same test set as your training run
dataset = dataset.train_test_split(test_size=0.2, seed=3407)
test_dataset = dataset["test"]

# 3. Load the pre-trained model from your local folder
print("Loading trained model...")
model = SetFitModel.from_pretrained("setfit-modernBERT-base-full-dataset")

# 4. Generate Predictions
print("Running predictions on the test set...")
# Note: Since we aren't using the Trainer's column_mapping here, we pass the "argument" column directly
X_test = test_dataset["argument"]
y_true = test_dataset["label"]

# Make predictions (this might take a minute depending on the test set size)
y_pred = model.predict(X_test)

# 5. Calculate and print metrics
print("\n" + "=" * 50)
print("             DETAILED EVALUATION REPORT")
print("=" * 50)

# Detailed per-class report
print("\n1. Per-Class Metrics:")
print(classification_report(y_true, y_pred, target_names=features, zero_division=0))

# Global summary metrics
micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
exact_accuracy = accuracy_score(y_true, y_pred)

print("\n2. Global Metrics Summary:")
print(f"   Micro F1 Score:      {micro_f1:.4f}  (Overall performance across all classes)")
print(f"   Macro F1 Score:      {macro_f1:.4f}  (Average performance per class)")
print(f"   Micro Precision:     {micro_precision:.4f}")
print(f"   Micro Recall:        {micro_recall:.4f}")
print(f"   Exact Match Accuracy:{exact_accuracy:.4f}  (Must predict ALL labels perfectly)")
print("=" * 50)

# Optional: Print a few sample predictions to see how it performs intuitively
print("\nSample Predictions:")
for i in range(3):
    text = X_test[i]
    # Convert binary arrays back to human readable labels
    true_labels = [features[idx] for idx, val in enumerate(y_true[i]) if val == 1.0]
    pred_labels = [features[idx] for idx, val in enumerate(y_pred[i]) if val == 1.0]

    print(f"\nText: {text}")
    print(f"True Labels:      {true_labels}")
    print(f"Predicted Labels: {pred_labels}")