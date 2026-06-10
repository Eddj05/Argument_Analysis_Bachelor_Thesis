import json
import numpy as np
import torch

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ============================================================
# 1. LOAD DATA
# ============================================================

with open("../data/JSON/annotated_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# remove invalid samples
data = [d for d in data if d.get("values")]

texts = [d["argument"] for d in data]
labels = [d["values"] for d in data]

print(f"Total samples: {len(texts)}")


# ============================================================
# 2. MULTI-LABEL ENCODING
# ============================================================

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

num_labels = len(mlb.classes_)

print("\nClasses:")
print(list(mlb.classes_))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    Y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. TOKENIZER
# ============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ============================================================
# 5. DATASET CLASS
# ============================================================

class ArgumentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.FloatTensor(self.labels[idx])

        return item


train_dataset = ArgumentDataset(X_train, y_train)
test_dataset  = ArgumentDataset(X_test, y_test)


# ============================================================
# 6. MODEL
# ============================================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)


# ============================================================
# 7. METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 🔥 IMPORTANT: convert logits → probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # 🔥 thresholding
    preds = (probs > 0.5).astype(int)

    # safety check (VERY useful for debugging)
    print("Pred label sum:", preds.sum())

    macro = f1_score(labels, preds, average="macro", zero_division=0)
    micro = f1_score(labels, preds, average="micro", zero_division=0)

    return {
        "macro_f1": macro,
        "micro_f1": micro
    }

# ============================================================
# 8. TRAINING CONFIG (BASELINE SETTINGS)
# ============================================================

training_args = TrainingArguments(
    output_dir="./bert_baseline",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none"
)


# ============================================================
# 9. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# ============================================================
# 10. TRAIN + EVALUATE
# ============================================================

trainer.train()

results = trainer.evaluate()

print("\n============================================================")
print("BERT BASELINE RESULTS")
print("============================================================")
print(results)