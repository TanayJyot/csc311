"""pred.py - CSC311 project prediction script.

Uses only the Python standard library, numpy and pandas. The models were trained
with scikit-learn in train_final.py; their parameters are read from model_params.npz
and model_meta.json, and the predictions are computed here with numpy.

Files needed next to this script: features.py, model_params.npz, model_meta.json.

Usage:
    from pred import predict_all
    predict_all("example_test_set.csv")
"""
import json
import os
import sys

import numpy as np

from features import (TEXT_COLS, design_matrix, load_survey,
                      structured_features, tokenize)

# Final model (Bernoulli naive Bayes); "random_forest" and "knn" are the other two
# model families compared in the report.
MODEL = "naive_bayes"
HERE = os.path.dirname(os.path.abspath(__file__))


def _load_model():
    with open(os.path.join(HERE, "model_meta.json")) as f:
        meta = json.load(f)
    with np.load(os.path.join(HERE, "model_params.npz")) as data:
        params = {k: data[k] for k in data.files}
    return meta, params


def _random_forest_proba(X, p):
    """Average of the class proportions in the leaf each row reaches in each tree."""
    X = X.astype(np.float32).astype(np.float64)    # sklearn compares float32 inputs
    rows = np.arange(len(X))
    proba = np.zeros((len(X), p["rf_value"].shape[1]))
    for root in p["rf_roots"]:
        node = np.full(len(X), root)
        inner = p["rf_left"][node] != -1
        while inner.any():
            n = node[inner]
            go_left = X[rows[inner], p["rf_feature"][n]] <= p["rf_threshold"][n]
            node[inner] = np.where(go_left, p["rf_left"][n], p["rf_right"][n])
            inner = p["rf_left"][node] != -1
        proba += p["rf_value"][node]
    return proba / len(p["rf_roots"])


def _naive_bayes_scores(X, p):
    """Bernoulli naive Bayes log-likelihood of each class, features binarised at 0."""
    B = (X > 0).astype(float)
    log_p = p["nb_feature_log_prob"]
    log_not_p = np.log1p(-np.exp(log_p))
    return B @ (log_p - log_not_p).T + log_not_p.sum(axis=1) + p["nb_class_log_prior"]


def _knn_scores(X, p, k):
    """Distance-weighted vote of the k nearest training rows (Euclidean distance)."""
    train = p["knn_X"]
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ train.T + (train ** 2).sum(1)[None, :]
    dist = np.sqrt(np.maximum(d2, 0))
    nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
    nd = np.take_along_axis(dist, nearest, axis=1)
    with np.errstate(divide="ignore"):
        w = 1.0 / nd
    exact = np.isinf(w)
    w[exact.any(axis=1)] = exact[exact.any(axis=1)]      # an exact match gets all the weight
    votes = np.zeros((len(X), int(p["knn_y"].max()) + 1))
    for j in range(k):
        np.add.at(votes, (np.arange(len(X)), p["knn_y"][nearest[:, j]]), w[:, j])
    return votes


def predict_all(filename, model=MODEL):
    """Predict 'Soccer', 'Baseball' or 'Basketball' for every row of the CSV file."""
    meta, params = _load_model()
    df = load_survey(filename)
    if len(df) == 0:
        return []
    struct = structured_features(df).drop(columns=["price"])
    tokens = {col: df[col].map(tokenize) for col in TEXT_COLS}
    X = design_matrix(struct, tokens, meta["preprocessing"][model])
    if model == "naive_bayes":
        scores = _naive_bayes_scores(X, params)
    elif model == "random_forest":
        scores = _random_forest_proba(X, params)
    else:
        scores = _knn_scores(X, params, meta["knn_k"])
    return [meta["classes"][i] for i in scores.argmax(axis=1)]


if __name__ == "__main__":
    print(predict_all(sys.argv[1])[:10])
