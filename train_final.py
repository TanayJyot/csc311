"""train_final.py - fit the three final models and export their parameters for pred.py.

Models (chosen in model_sweep/): random forest, Bernoulli naive Bayes and L1 softmax
regression. Each is fit on all labelled rows, and its parameters are saved to
model_params.npz (arrays) and model_meta.json (vocabularies and column order), which
pred.py reads with numpy and pandas only.

Run: python train_final.py   (needs scikit-learn; writes the two files next to pred.py)
check_pred.py calls fit_and_export(holdout=True) to test pred.py on held-out rows.
"""
import json
import os
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from features import (TEXT_COLS, blank_rows, design_matrix, load_survey,
                      respondent_split, structured_features, tokenize)

DATA = "csc311_project_training_202609.csv"
MIN_DF = {"random_forest": 1, "bernoulli_nb": 1, "softmax": 2}


def fit_preprocessing(struct, tokens, min_df):
    """Imputation medians, scaling statistics and per-question vocabularies."""
    filled = struct.fillna(struct.median())
    vocab = {}
    for col in TEXT_COLS:
        counts = Counter(t for toks in tokens[col] for t in set(toks))
        vocab[col] = sorted(t for t, n in counts.items() if n >= min_df)
    return {"columns": list(struct.columns),
            "median": struct.median().tolist(),
            "mean": filled.mean().tolist(),
            "std": filled.std().replace(0, 1).tolist(),
            "vocab": vocab}


def fit_and_export(out_dir=".", holdout=False):
    """Fit the three models and write model_params.npz and model_meta.json to out_dir.

    holdout=True fits on the instructors' train+val respondents only.
    """
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    if holdout:
        df = df[respondent_split(df["id"]) != "test"].reset_index(drop=True)
    y = df["label"].to_numpy()
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}

    preps = {m: fit_preprocessing(struct, tokens, k) for m, k in MIN_DF.items()}
    X = {m: design_matrix(struct, tokens, p) for m, p in preps.items()}

    rf = RandomForestClassifier(500, max_features="log2", random_state=0).fit(X["random_forest"], y)
    nb = BernoulliNB(alpha=0.5, binarize=0.0).fit(X["bernoulli_nb"], y)
    sm = LogisticRegression(C=0.5, l1_ratio=1.0, solver="saga", max_iter=5000).fit(X["softmax"], y)
    assert list(rf.classes_) == list(nb.classes_) == list(sm.classes_)

    # Random forest: all trees in flat arrays; node ids are offset per tree.
    left, right, feat, thr, val, roots = [], [], [], [], [], []
    offset = 0
    for est in rf.estimators_:
        t = est.tree_
        leaf = t.children_left == -1
        left.append(np.where(leaf, -1, t.children_left + offset))
        right.append(np.where(leaf, -1, t.children_right + offset))
        feat.append(np.where(leaf, 0, t.feature))
        thr.append(t.threshold)
        v = t.value[:, 0, :]
        val.append(v / v.sum(1, keepdims=True))
        roots.append(offset)
        offset += t.node_count

    np.savez_compressed(
        os.path.join(out_dir, "model_params.npz"),
        rf_left=np.concatenate(left).astype(np.int32),
        rf_right=np.concatenate(right).astype(np.int32),
        rf_feature=np.concatenate(feat).astype(np.int32),
        rf_threshold=np.concatenate(thr),
        rf_value=np.concatenate(val).astype(np.float32),
        rf_roots=np.array(roots, dtype=np.int32),
        nb_feature_log_prob=nb.feature_log_prob_,
        nb_class_log_prior=nb.class_log_prior_,
        sm_coef=sm.coef_,
        sm_intercept=sm.intercept_,
    )
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump({"classes": list(rf.classes_), "preprocessing": preps}, f)

    print(f"trained on {len(df)} rows; random forest has {offset} nodes")
    return {"random_forest": rf, "bernoulli_nb": nb, "softmax": sm}, preps


def main():
    fit_and_export()


if __name__ == "__main__":
    main()
