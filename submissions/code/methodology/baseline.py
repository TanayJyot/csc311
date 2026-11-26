"""baseline.py - pipeline sanity check quoted in the proposal (training split only).

UNTUNED versions of the four planned model families (kNN, decision tree, softmax
regression, MLP), scored with grouped 5-fold cross-validation on the training split.
These show the pipeline runs end to end; they are not model-selection results.

Also measures the optional "joint assignment" idea: each respondent answered once
per sport, so their three rows can be given the label permutation with the highest
total log-probability.

Run: python baseline.py   (needs scikit-learn in addition to numpy and pandas)
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "methodology"))   # features.py

from features import (LABELS, LIKERT_COLS, TEXT_COLS, blank_rows, load_survey,
                      clean_survey, dev_test_split, structured_features, tokenize)

DATA = str(ROOT / "csc311_project_training_202609.csv")


def design_matrix(struct, tokens, fit_idx, min_df=2):
    """Impute/scale with statistics from fit_idx only; binary bag of words per question."""
    X = struct.copy()
    flags = X[["ease", "log_price"] + LIKERT_COLS].isna().astype(float).add_suffix("_missing")
    X = X.fillna(X.iloc[fit_idx].median())
    mu, sd = X.iloc[fit_idx].mean(), X.iloc[fit_idx].std().replace(0, 1)
    parts = [((X - mu) / sd).to_numpy(), flags.to_numpy()]
    for col in TEXT_COLS:
        df_count = Counter(t for i in fit_idx for t in set(tokens[col].iloc[i]))
        vocab = {t: j for j, t in enumerate(sorted(t for t, n in df_count.items() if n >= min_df))}
        M = np.zeros((len(X), len(vocab) + 1))
        for i, toks in enumerate(tokens[col]):
            for t in set(toks):
                if t in vocab:
                    M[i, vocab[t]] = 1.0
            M[i, -1] = float(len(toks) == 0)            # missing / non-answer indicator
        parts.append(M)
    return np.hstack(parts)


def main():
    df = load_survey(DATA)
    df = clean_survey(df)
    df["split"] = dev_test_split(df["id"])
    tr = df[df["split"] == "dev"].reset_index(drop=True)
    y, groups = tr["label"].to_numpy(), tr["id"].to_numpy()
    struct = structured_features(tr).drop(columns=["price"])
    tokens = {col: tr[col].map(tokenize) for col in TEXT_COLS}

    folds = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups))
    untuned = {
        "kNN (k = 15)": lambda: KNeighborsClassifier(15),
        "decision tree (entropy, depth 8, leaf 5)": lambda: DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=5, random_state=0),
        "softmax regression (C = 1)": lambda: LogisticRegression(C=1.0, max_iter=5000),
        "MLP (32 ReLU units, mini-batch SGD)": lambda: MLPClassifier(
            (32,), solver="sgd", learning_rate_init=0.05, alpha=1e-2, batch_size=32,
            max_iter=300, early_stopping=True, random_state=0),
    }
    for name, make in untuned.items():
        accs = []
        for fit_idx, held_idx in folds:
            X = design_matrix(struct, tokens, fit_idx, min_df=3)
            model = make().fit(X[fit_idx], y[fit_idx])
            accs.append((model.predict(X[held_idx]) == y[held_idx]).mean())
        print(f"{name}: {100 * np.mean(accs):.1f}% +/- {100 * np.std(accs):.1f}%")
    print()

    accs, oof = [], np.zeros((len(tr), 3))
    for fit_idx, held_idx in GroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups):
        X = design_matrix(struct, tokens, fit_idx, min_df=3)
        model = LogisticRegression(C=1.0, max_iter=5000).fit(X[fit_idx], y[fit_idx])
        oof[held_idx] = model.predict_proba(X[held_idx])
        accs.append((model.classes_[oof[held_idx].argmax(1)] == y[held_idx]).mean())
    classes = model.classes_
    pred = classes[oof.argmax(1)]
    print(f"untuned softmax regression: grouped 5-fold CV accuracy "
          f"{100 * np.mean(accs):.1f}% +/- {100 * np.std(accs):.1f}% (folds: "
          f"{', '.join(f'{100 * a:.1f}' for a in accs)})")
    print(f"macro-F1 {f1_score(y, pred, average='macro'):.3f}")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS))

    joint = pred.copy()
    for _, idx in tr.groupby("id").indices.items():
        if len(idx) == 3:
            best = max(itertools.permutations(range(3)),
                       key=lambda p: sum(np.log(oof[idx[k], p[k]] + 1e-12) for k in range(3)))
            joint[idx] = classes[list(best)]
    print(f"with joint assignment per respondent: {100 * (joint == y).mean():.1f}%")


if __name__ == "__main__":
    main()
