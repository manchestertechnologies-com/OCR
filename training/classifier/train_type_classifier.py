"""Train the question-type classifier used for ambiguous cases.

Character n-gram TF-IDF + logistic regression: robust to OCR slips ("Assertlon") and fast
on CPU. Uses text_edited when a reviewer fixed it, else the OCR text. Only questions whose
labels a reviewer confirmed (labels_reviewed) or that were not flagged are used.

  python training/classifier/train_type_classifier.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examocr import store  # noqa: E402


def load_examples():
    X, y = [], []
    for meta in store.list_projects():
        if meta["status"] != "ready":
            continue
        for q in store.load(meta["id"]).questions:
            if not (q.labels_reviewed or not q.needs_review):
                continue
            X.append(q.text_edited or q.text)
            y.append(q.question_type.value)
    return X, y


def main():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline

    X, y = load_examples()
    counts = Counter(y)
    print("examples per type:", dict(counts))
    if len(X) < 50 or len(counts) < 2:
        sys.exit("Need at least ~50 labelled questions across 2+ types. Review more papers first.")

    clf = Pipeline([
        ("feats", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                                     min_df=2, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                     min_df=1, sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    strat = y if min(counts.values()) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=strat)
    clf.fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte), zero_division=0))
    clf.fit(X, y)  # final model on everything
    out = ROOT / "models/qtype_clf.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(clf, out)
    print("saved", out)


if __name__ == "__main__":
    main()
