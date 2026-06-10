import json
import numpy as np
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.exceptions import UndefinedMetricWarning


# =========================
# SUPPRESS WARNINGS (CLEAN OUTPUT)
# =========================

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# =========================
# LOAD + CLEAN DATA
# =========================

with open("../data/JSON/annotated_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data = [item for item in data if item.get("values")]

texts = [item["argument"] for item in data]
labels = [item["values"] for item in data]


# =========================
# MULTI-LABEL ENCODING
# =========================

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)


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
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=3000, class_weight="balanced")
        ))
    ])


def build_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", OneVsRestClassifier(
            LinearSVC(class_weight="balanced")
        ))
    ])


# =========================
# EVALUATION
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

        y_train = Y[train_idx]
        y_test  = Y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # =========================
        # FIX: zero_division=0
        # =========================
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        micro = f1_score(y_test, preds, average="micro", zero_division=0)

        macro_scores.append(macro)
        micro_scores.append(micro)

        print(f"Fold {fold+1} | Macro={macro:.4f} Micro={micro:.4f}")

    print("\nFINAL RESULTS")
    print(f"Macro F1: {np.mean(macro_scores):.4f} ± {np.std(macro_scores):.4f}")
    print(f"Micro F1: {np.mean(micro_scores):.4f} ± {np.std(micro_scores):.4f}")


# =========================
# RUN MODELS
# =========================

run(build_lr, "MULTI-LABEL TF-IDF + Logistic Regression")
run(build_svm, "MULTI-LABEL TF-IDF + Linear SVM")