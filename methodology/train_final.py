"""train_final.py - fit the three final models and export their parameters for pred.py.

The settings are read from model_selection_results/summary.csv, the tuned ones from
model_selection.py. Following the report, each
model is fit on the 70% development set; the test rows are kept out so the
reported test accuracy is that of this exact model.

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

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "methodology" / "submission"))   # features.py

from features import (TEXT_COLS, clean_survey, design_matrix, dev_test_split,
                      fit_preprocessing, load_survey, structured_features, tokenize)

DATA = str(ROOT / "csc311_project_training_202609.csv")
RESULTS = HERE / "model_selection_results" / "summary.csv"


def tuned_settings():
    """The best setting of each family, exactly as model_selection.py chose it."""
    import pandas as pd
    S = pd.read_csv(RESULTS).set_index("family")
    return {"naive_bayes": json.loads(S.loc["Naive Bayes", "params"]),
            "random_forest": json.loads(S.loc["Random forest", "params"]),
            "knn": json.loads(S.loc["kNN", "params"])}


def fit_and_export(out_dir=str(HERE.parent / "methodology" / "submission")):
    df = load_survey(DATA)
    df = clean_survey(df)
    df = df[dev_test_split(df["id"]) == "dev"].reset_index(drop=True)
    y = df["label"].to_numpy()
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}

    P = tuned_settings()
    settings = {m: dict(scaling=p.get("scaling", "z-score"), words=p.get("words", "all"))
                for m, p in P.items()}
    preps = {m: fit_preprocessing(struct, tokens, y, **s) for m, s in settings.items()}
    X = {m: design_matrix(struct, tokens, p) for m, p in preps.items()}

    nb = BernoulliNB(alpha=P["naive_bayes"]["alpha"], binarize=0.0).fit(X["naive_bayes"], y)
    r = P["random_forest"]
    rf = RandomForestClassifier(r["n_trees"], criterion=r["criterion"], max_features=r["max_features"],
                                max_depth=r["max_depth"], min_samples_leaf=r["min_leaf"],
                                random_state=0).fit(X["random_forest"], y)
    k = P["knn"]
    knn = KNeighborsClassifier(k["k"], weights=k["weights"]).fit(X["knn"], y)
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
        json.dump({"classes": classes, "knn_k": k["k"], "knn_weights": k["weights"], "preprocessing": preps}, f)
    print(f"trained on {len(df)} rows; random forest has {offset} nodes")
    return {"naive_bayes": nb, "random_forest": rf, "knn": knn}, preps


if __name__ == "__main__":
    fit_and_export()
