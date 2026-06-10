from datasets import load_dataset
from setfit import SetFitModel
from setfit import Trainer
from setfit import TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# 1. Update Metrics for Multi-Label
# In multi-label, precision, recall, and F1 require an 'average' parameter.
def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)  # Note: exact match accuracy
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    return {'accuracy': accuracy, 'macro_precision': macro_precision, 'macro_recall': macro_recall,
            'macro_f1': macro_f1, 'micro_precision': micro_precision, 'micro_recall': micro_recall,
            'micro_f1': micro_f1}


features = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement',
            'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition',
            'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring',
            'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
            'Universalism: objectivity']

# 2. Tell SetFit this is a Multi-Label task
model = SetFitModel.from_pretrained(
    "answerdotai/ModernBERT-base",
    multi_target_strategy="one-vs-rest"  # Enables Multi-Label
)
model.labels = features

dataset = load_dataset("json", data_files="complete_dataset_test.json", split="train")


# 3. Create a function to Multi-Hot Encode the labels
def encode_labels(record):
    # Initialize an array of 0s with length 20
    encoded = [0.0] * len(features)
    # Put a 1.0 at the index of the labels present in the record
    for val in record["values"]:
        if val in features:
            encoded[features.index(val)] = 1.0
    # Map it directly to the 'label' column
    return {"label": encoded}


# Apply the encoding to the dataset
dataset = dataset.map(encode_labels)

# Split dataset after encoding
dataset = dataset.train_test_split(test_size=0.2, seed=3407)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

args = TrainingArguments(
    batch_size=2,
    num_epochs=5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # We remove "values": "label" from mapping because we already created the "label" column via map()
    column_mapping={"argument": "text"},
    metric=compute_metrics
)

trainer.train()

# Evaluate and print results
print(trainer.evaluate(test_dataset))

model.save_pretrained("setfit-modernBERT-large-full-dataset")
