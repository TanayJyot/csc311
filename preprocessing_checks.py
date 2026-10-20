"""preprocessing_checks.py - evidence for the "Feature types, missing values and
preprocessing" section (outliers, encoding, scaling), on the training split only.

Part A prints the descriptive numbers the section quotes (feature ranges, price
outliers, words per answer, unseen words, feature counts). Part B re-runs the
untuned softmax regression from baseline.py (same grouped 5-fold CV, same
design matrix) while changing one preprocessing choice at a time, so each
choice in the report is backed by a measured accuracy difference.

Run:  python preprocessing_checks.py
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from baseline import DATA, design_matrix
from features import (LIKERT_COLS, MULTI_OPTIONS, TEXT_COLS, blank_rows, load_survey,
                      parse_price, respondent_split, structured_features, tokenize)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def cv_accuracy(make_X, y, folds, make_model):
    """Mean and SD of held-out accuracy; make_X(fit_idx) builds the full matrix."""
    accs = []
    for fit_idx, held_idx in folds:
        X = make_X(fit_idx)
        model = make_model().fit(X[fit_idx], y[fit_idx])
        accs.append((model.predict(X[held_idx]) == y[held_idx]).mean())
    return 100 * np.mean(accs), 100 * np.std(accs)


# ----------------------------------------------------------------------------
# Variants of baseline.design_matrix, each changing exactly one choice
# ----------------------------------------------------------------------------
def word_block(tokens, fit_idx, n_rows, min_df=3, weighting="binary"):
    """Per-question bag of words (vocabulary from fit_idx) + a no-token indicator."""
    parts = []
    for col in TEXT_COLS:
        df_count = Counter(t for i in fit_idx for t in set(tokens[col].iloc[i]))
        vocab = {t: j for j, t in enumerate(sorted(t for t, n in df_count.items() if n >= min_df))}
        idf = np.log(len(fit_idx) / np.array([df_count[t] for t in sorted(vocab, key=vocab.get)]))
        M = np.zeros((n_rows, len(vocab) + 1))
        for i, toks in enumerate(tokens[col]):
            for t in toks:
                if t in vocab:
                    M[i, vocab[t]] = 1.0 if weighting == "binary" else M[i, vocab[t]] + 1.0
            M[i, -1] = float(len(toks) == 0)
        if weighting == "tfidf":
            M[:, :-1] *= idf
            norms = np.linalg.norm(M[:, :-1], axis=1, keepdims=True)
            M[:, :-1] /= np.where(norms == 0, 1, norms)
        parts.append(M)
    return np.hstack(parts)


def variant_matrix(struct, tokens, fit_idx, likert="ordinal", scaling="z", scale_cols="struct",
                   weighting="binary", clip_price=False, flags=True):
    X = struct.copy()
    if clip_price:
        X["log_price"] = X["log_price"].clip(upper=X["log_price"].iloc[fit_idx].quantile(0.99))
    flag_cols = ["ease", "log_price"] + LIKERT_COLS
    F = X[flag_cols].isna().astype(float).to_numpy()
    X = X.fillna(X.iloc[fit_idx].median())
    if likert == "onehot":
        for col in LIKERT_COLS:
            for level in range(1, 6):
                X[f"{col}={level}"] = (X[col].round() == level).astype(float)
            X = X.drop(columns=col)
    if scaling == "z":
        mu, sd = X.iloc[fit_idx].mean(), X.iloc[fit_idx].std().replace(0, 1)
        S = ((X - mu) / sd).to_numpy()
    elif scaling == "minmax":
        lo, hi = X.iloc[fit_idx].min(), X.iloc[fit_idx].max()
        S = ((X - lo) / (hi - lo).replace(0, 1)).to_numpy()
    else:
        S = X.to_numpy(float)
    W = word_block(tokens, fit_idx, len(X), weighting=weighting)
    if scale_cols == "all" and scaling == "z":
        mu, sd = W[fit_idx].mean(0), W[fit_idx].std(0)
        W = (W - mu) / np.where(sd == 0, 1, sd)
    parts = [S] + ([F] if flags else []) + [W]
    return np.hstack(parts)


def main():
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    df["split"] = respondent_split(df["id"])
    tr = df[df["split"] == "train"].reset_index(drop=True)
    y, groups = tr["label"].to_numpy(), tr["id"].to_numpy()
    raw = structured_features(tr)
    struct = raw.drop(columns=["price"])
    tokens = {col: tr[col].map(tokenize) for col in TEXT_COLS}
    folds = list(GroupKFold(5, shuffle=True, random_state=0).split(tr, y, groups))

    # ------------------------------------------------------------------ Part A
    section("A1. Feature ranges before scaling (training split)")
    rng = struct[["ease"] + LIKERT_COLS + ["log_price"]].agg(["min", "max", "mean", "std"]).T
    print(rng.round(2).to_string())
    print(f"ease values that are not whole numbers: "
          f"{int((struct['ease'].dropna() % 1 != 0).sum())}")
    n_multi = sum(len(v) for v in MULTI_OPTIONS.values())
    print(f"multi-hot columns: {n_multi} ({', '.join(f'{k} {len(v)}' for k, v in MULTI_OPTIONS.items())}), values 0/1")

    section("A2. Price outliers (Q6)")
    p = raw["price"].dropna()
    lp = np.log1p(p)
    q1, q3 = lp.quantile([.25, .75])
    fence = q3 + 1.5 * (q3 - q1)
    low_fence = q1 - 1.5 * (q3 - q1)
    print(f"raw $: median {p.median():.0f}, mean {p.mean():.0f}, max {p.max():.0f}, "
          f"skewness {p.skew():.2f}; log1p skewness {lp.skew():.2f}")
    print(f"log1p range {lp.min():.2f}-{lp.max():.2f}; upper fence {fence:.3f} (= ${np.expm1(fence):,.0f}), "
          f"lower fence {low_fence:.3f} (= ${max(np.expm1(low_fence), 0):,.2f})")
    print(f"above upper fence: {int((lp > fence).sum())} -> {sorted(p[lp > fence].astype(int).tolist())}")
    print(f"below lower fence: {int((lp < low_fence).sum())} (these are the $0 refusals: "
          f"{int((p == 0).sum())} zeros)")
    z_max = (lp.max() - lp.mean()) / lp.std()
    print(f"largest price after log1p is {z_max:.2f} SD above the mean "
          f"(raw scale: {(p.max() - p.mean()) / p.std():.2f} SD)")
    print(f"99th percentile ${p.quantile(.99):,.0f}; values above it: "
          f"{sorted(p[p > p.quantile(.99)].astype(int).tolist())}")
    by_sport = raw.assign(label=y).loc[raw['price'].notna()]
    print("rows above the upper fence by sport:",
          by_sport.loc[np.log1p(by_sport['price']) > fence, 'label'].value_counts().to_dict())

    section("A3. Price parsing: what counts as unparseable (Q7c)")
    unparsed = tr.loc[raw["price"].isna(), "price"]
    scale = unparsed.str.lower().str.contains("scale")
    print(f"unparseable: {len(unparsed)} of {len(tr)} ({100 * len(unparsed) / len(tr):.1f}%); "
          f"'1 to 10 scale' answers: {int(scale.sum())}; no amount found: {int((~scale).sum())}")
    cur = tr["price"].str.lower().str.contains(r"cad|usd|us\$|rmb|yuan|€|£|¥|eur|元", regex=True)
    print(f"answers naming a currency: {int(cur.sum())} (amounts are used as-is, not converted)")

    section("A4. Free text: words per answer, repeats, unseen words, feature count (Q7d-g)")
    for col in TEXT_COLS:
        toks = tokens[col]
        answered = toks[toks.map(len) > 0]
        rep = answered.map(lambda t: len(t) != len(set(t)))
        print(f"{col:8s} mean tokens per answered row {answered.map(len).mean():.2f}; "
              f"answers with a repeated word "
              f"{100 * rep.mean():.1f}%")
    all_tok = [t for col in TEXT_COLS for toks in tokens[col] for t in toks]
    rows_rep = sum(len(t) != len(set(t)) for col in TEXT_COLS for t in tokens[col])
    print(f"over all four questions: {100 * rows_rep / (4 * len(tr)):.1f}% of answers repeat a word")
    unseen_tok, total_tok, unseen_rows = 0, 0, 0
    for fit_idx, held_idx in folds:
        vocab = {col: {t for i in fit_idx for t in tokens[col].iloc[i]} for col in TEXT_COLS}
        for i in held_idx:
            row_unseen = False
            for col in TEXT_COLS:
                for t in tokens[col].iloc[i]:
                    total_tok += 1
                    if t not in vocab[col]:
                        unseen_tok += 1
                        row_unseen = True
            unseen_rows += row_unseen
    print(f"(baseline.py folds) held-out tokens never seen in the training folds: {100 * unseen_tok / total_tok:.1f}%; "
          f"held-out rows with at least one: {100 * unseen_rows / len(tr):.1f}%")
    vocab1 = sum(len({t for toks in tokens[c] for t in toks}) for c in TEXT_COLS)
    vocab3 = sum(sum(1 for n in Counter(t for toks in tokens[c] for t in set(toks)).values() if n >= 3)
                 for c in TEXT_COLS)
    n_struct = struct.shape[1]
    print(f"structured columns: {n_struct}; words (min_df 1): {vocab1}; words (min_df 3): {vocab3}")
    print(f"all words: {n_struct + vocab1} features for {len(tr)} rows; min_df 3: {n_struct + vocab3}")
    widths = [design_matrix(struct, tokens, f, min_df=3).shape[1] for f, _ in folds]
    print(f"baseline.py design matrix width per fold (min_df 3, + 5 flags + 4 text indicators): {widths}")

    # ------------------------------------------------------------------ Part B
    section("B. One change at a time: untuned softmax regression (C = 1), grouped 5-fold CV")
    logreg = lambda: LogisticRegression(C=1.0, max_iter=5000)
    knn = lambda: KNeighborsClassifier(15)
    base_check = cv_accuracy(lambda f: design_matrix(struct, tokens, f, min_df=3), y, folds, logreg)
    print(f"{'baseline.py design_matrix (reference)':52s} {base_check[0]:.1f}% +/- {base_check[1]:.1f}")
    variants = [
        ("as baseline (variant_matrix defaults)", {}),
        ("Likert one-hot instead of ordinal", {"likert": "onehot"}),
        ("word counts instead of binary", {"weighting": "count"}),
        ("TF-IDF (L2-normalised) instead of binary", {"weighting": "tfidf"}),
        ("clip log price at training 99th percentile", {"clip_price": True}),
        ("no missing flags for numeric/Likert/price", {"flags": False}),
        ("min-max scaling instead of z-score", {"scaling": "minmax"}),
        ("no scaling", {"scaling": "none"}),
        ("z-score the word columns too", {"scale_cols": "all"}),
    ]
    for name, kw in variants:
        m, s = cv_accuracy(lambda f, kw=kw: variant_matrix(struct, tokens, f, **kw), y, folds, logreg)
        print(f"{name:52s} {m:.1f}% +/- {s:.1f}")

    print("\nkNN (k = 15), same folds: does scaling matter?")
    for name, kw in [("z-score (baseline)", {}), ("min-max", {"scaling": "minmax"}),
                     ("no scaling", {"scaling": "none"})]:
        m, s = cv_accuracy(lambda f, kw=kw: variant_matrix(struct, tokens, f, **kw), y, folds, knn)
        print(f"{name:52s} {m:.1f}% +/- {s:.1f}")


if __name__ == "__main__":
    main()
