"""train_final.py - fit the three final models and export their parameters for pred.py.

The settings are the tuned ones from model_selection.py. Following the report, each
model is fit on the instructors' training + validation respondents (609 rows); the
254 test rows are kept out so the reported test accuracy is that of this exact model.

Writes model_params.npz (arrays) and model_meta.json (preprocessing, vocabularies),
which pred.py reads with numpy and pandas only.
Run: python train_final.py   (needs scikit-learn)
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from features import (TEXT_COLS, blank_rows, design_matrix, fit_preprocessing,
                      load_survey, respondent_split, structured_features, tokenize)

DATA = "csc311_project_training_202609.csv"
SETTINGS = {
    "naive_bayes": dict(scaling="z-score", words=50),
    "random_forest": dict(scaling="z-score", words="all"),
    "knn": dict(scaling="min-max", words=50),
}


def fit_and_export(out_dir="."):
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    df = df[respondent_split(df["id"]) != "test"].reset_index(drop=True)
    y = df["label"].to_numpy()
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}

    preps = {m: fit_preprocessing(struct, tokens, y, **s) for m, s in SETTINGS.items()}
    X = {m: design_matrix(struct, tokens, p) for m, p in preps.items()}

    nb = BernoulliNB(alpha=0.5, binarize=0.0).fit(X["naive_bayes"], y)
    rf = RandomForestClassifier(500, criterion="entropy", max_features="log2",
                                random_state=0).fit(X["random_forest"], y)
    knn = KNeighborsClassifier(7, weights="distance").fit(X["knn"], y)
    assert list(nb.classes_) == list(rf.classes_) == list(knn.classes_)

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

    classes = list(nb.classes_)
    np.savez_compressed(
        os.path.join(out_dir, "model_params.npz"),
        nb_feature_log_prob=nb.feature_log_prob_,
        nb_class_log_prior=nb.class_log_prior_,
        rf_left=np.concatenate(left).astype(np.int32),
        rf_right=np.concatenate(right).astype(np.int32),
        rf_feature=np.concatenate(feat).astype(np.int32),
        rf_threshold=np.concatenate(thr),
        rf_value=np.concatenate(val).astype(np.float32),
        rf_roots=np.array(roots, dtype=np.int32),
        knn_X=X["knn"],
        knn_y=np.array([classes.index(c) for c in y], dtype=np.int8),
    )
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump({"classes": classes, "knn_k": 7, "preprocessing": preps}, f)
    print(f"trained on {len(df)} rows; random forest has {offset} nodes")
    return {"naive_bayes": nb, "random_forest": rf, "knn": knn}, preps


if __name__ == "__main__":
    fit_and_export()
