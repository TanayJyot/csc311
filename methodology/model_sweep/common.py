"""Shared data loading, design matrices and evaluation for the model sweep."""
import itertools, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from features import (LIKERT_COLS, TEXT_COLS, blank_rows, load_survey,
                      respondent_split, structured_features, tokenize)

DATA = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "csc311_project_training_202609.csv")

def load_all():
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    df["split"] = respondent_split(df["id"])
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}
    return df, struct, tokens

def design(struct, tokens, fit_idx, min_df=2, text="binary", use_text=True):
    """Fit preprocessing on fit_idx only. text in {binary, tfidf}."""
    X = struct.copy()
    flags = X[["ease", "log_price"] + LIKERT_COLS].isna().astype(float).add_suffix("_missing")
    X = X.fillna(X.iloc[fit_idx].median())
    mu, sd = X.iloc[fit_idx].mean(), X.iloc[fit_idx].std().replace(0, 1)
    parts = [((X - mu) / sd).to_numpy(), flags.to_numpy()]
    if use_text:
        n_fit = len(fit_idx)
        for col in TEXT_COLS:
            dfc = Counter(t for i in fit_idx for t in set(tokens[col].iloc[i]))
            vocab = {t: j for j, t in enumerate(sorted(t for t, n in dfc.items() if n >= min_df))}
            M = np.zeros((len(X), len(vocab) + 1))
            for i, toks in enumerate(tokens[col]):
                for t in set(toks):
                    if t in vocab:
                        M[i, vocab[t]] = 1.0
                M[i, -1] = float(len(toks) == 0)
            if text == "tfidf":
                idf = np.ones(M.shape[1])
                idf[:-1] = np.log((1 + n_fit) / (1 + np.array([dfc[t] for t in sorted(vocab, key=vocab.get)]))) + 1
                M = M * idf
                nrm = np.linalg.norm(M[:, :-1], axis=1, keepdims=True); nrm[nrm == 0] = 1
                M[:, :-1] /= nrm
            parts.append(M)
    return np.hstack(parts)

def joint_assign(proba, ids):
    """Each respondent with exactly 3 rows gets 3 distinct labels (best total log-prob)."""
    pred = proba.argmax(1).copy()
    for _, idx in pd.Series(ids).groupby(ids).indices.items():
        if len(idx) == 3:
            best = max(itertools.permutations(range(3)),
                       key=lambda p: sum(np.log(proba[idx[k], p[k]] + 1e-12) for k in range(3)))
            pred[idx] = list(best)
    return pred

def scores_to_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass
    d = model.decision_function(X)
    d = d - d.max(1, keepdims=True)
    e = np.exp(d)
    return e / e.sum(1, keepdims=True)
