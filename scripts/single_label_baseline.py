import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# =========================
# LOAD data
# =========================

with open("../data/JSON/annotated_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# keep only valid samples
data = [item for item in data if item.get("values")]

texts = []
labels = []

for item in data:
    texts.append(item["argument"])
    labels.append(item["values"][0])  # single-label = first value


print(f"Total samples after cleaning: {len(texts)}")


# =========================
# CROSS-VALIDATION
# =========================

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# =========================
# MODELS
# =========================

def build_lr():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])


def build_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])


# =========================
# EVALUATION FUNCTION
# =========================

def run(model_fn, name):

    macro_scores = []
    micro_scores = []

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(cv.split(texts)):

        X_train = [texts[i] for i in train_idx]
        X_test  = [texts[i] for i in test_idx]

        y_train = [labels[i] for i in train_idx]
        y_test  = [labels[i] for i in test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        macro = f1_score(y_test, preds, average="macro")
        micro = f1_score(y_test, preds, average="micro")

        macro_scores.append(macro)
        micro_scores.append(micro)

        print(f"Fold {fold+1} | Macro={macro:.4f} Micro={micro:.4f}")

    print("\nFINAL RESULTS")
    print(f"Macro F1: {np.mean(macro_scores):.4f} ± {np.std(macro_scores):.4f}")
    print(f"Micro F1: {np.mean(micro_scores):.4f} ± {np.std(micro_scores):.4f}")


# =========================
# RUN MODELS
# =========================

run(build_lr, "SINGLE-LABEL TF-IDF + Logistic Regression")
run(build_svm, "SINGLE-LABEL TF-IDF + Linear SVM")