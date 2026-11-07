"""model_selection.py - the model selection reported in the final report.

Protocol (Methodology section):
1. Tune kNN, random forest and Bernoulli naive Bayes with grouped 5-fold CV on the
   instructors' training split (146 respondents, 438 rows); preprocessing is fit on
   the four training folds each time.
2. Compare the three tuned models on the validation split (171 rows): accuracy,
   per-sport F1 and the confusion matrix.
3. Retrain the chosen model on train + validation (609 rows) and score it once on
   the test split (254 rows).

Run: python model_selection.py   (needs scikit-learn; writes model_selection_results/)
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from features import (LABELS, TEXT_COLS, blank_rows, design_matrix,
                      fit_preprocessing, load_survey, respondent_split,
                      structured_features, tokenize)

DATA = "csc311_project_training_202609.csv"
OUT = "model_selection_results"
MIN_DF = 3                      # keep words used by at least 3 training rows


def grid(**values):
    return [dict(zip(values, combo)) for combo in itertools.product(*values.values())]


GRIDS = {
    "kNN": grid(k=[1, 3, 5, 7, 11, 15, 21, 31, 51], words=["all", 25, 50, 100],
                weights=["uniform", "distance"], scaling=["z-score", "min-max"]),
    "Random forest": grid(n_trees=[100, 300, 500], max_features=["sqrt", "log2", 0.1, 0.3],
                          max_depth=[None, 10, 20], min_leaf=[1, 2, 5],
                          criterion=["gini", "entropy"]),
    "Naive Bayes": grid(alpha=[0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10],
                        words=["all", 25, 50, 100]),
}


def make_model(family, p):
    if family == "kNN":
        return KNeighborsClassifier(p["k"], weights=p["weights"])
    if family == "Random forest":
        return RandomForestClassifier(p["n_trees"], max_features=p["max_features"],
                                      max_depth=p["max_depth"], min_samples_leaf=p["min_leaf"],
                                      criterion=p["criterion"], random_state=0)
    return BernoulliNB(alpha=p["alpha"], binarize=0.0)


def design(struct, tokens, fit, y, scaling="z-score", words="all"):
    """Feature matrix for all rows, with every statistic learned from the rows in `fit`."""
    sub = {c: tokens[c].iloc[fit].reset_index(drop=True) for c in TEXT_COLS}
    prep = fit_preprocessing(struct.iloc[fit], sub, y[fit], scaling, words, MIN_DF)
    return design_matrix(struct, tokens, prep)


def per_sport_f1(y_true, y_pred):
    return dict(zip(LABELS, f1_score(y_true, y_pred, labels=LABELS, average=None)))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    df["split"] = respondent_split(df["id"])
    y, ids = df["label"].to_numpy(), df["id"].to_numpy()
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}
    train = np.where(df.split == "train")[0]
    val = np.where(df.split == "val")[0]
    test = np.where(df.split == "test")[0]
    folds = [(train[a], train[b]) for a, b in
             GroupKFold(5, shuffle=True, random_state=0).split(train, groups=ids[train])]

    # 1. Grouped 5-fold CV on the training split. Design matrices are cached per
    #    (fold, scaling, words) because they do not depend on the model.
    cache = {}

    def X_for(fold_i, scaling, words):
        key = (fold_i, scaling, words)
        if key not in cache:
            fit = folds[fold_i][0] if fold_i is not None else None
            cache[key] = design(struct, tokens, fit, y, scaling, words)
        return cache[key]

    for i in range(len(folds)):
        for s in ("z-score", "min-max"):
            for w in ("all", 25, 50, 100):
                X_for(i, s, w)

    def run(family, p):
        accs, train_accs = [], []
        for i, (fit, held) in enumerate(folds):
            X = X_for(i, p.get("scaling", "z-score"), p.get("words", "all"))
            m = make_model(family, p).fit(X[fit], y[fit])
            accs.append(np.mean(m.predict(X[held]) == y[held]))
            train_accs.append(np.mean(m.predict(X[fit]) == y[fit]))
        return dict(family=family, params=json.dumps(p), cv_acc=np.mean(accs),
                    cv_sd=np.std(accs), train_acc=np.mean(train_accs))

    jobs = [(f, p) for f, g in GRIDS.items() for p in g]
    cv = pd.DataFrame(Parallel(n_jobs=4)(delayed(run)(f, p) for f, p in jobs))
    cv.to_csv(f"{OUT}/cv_all.csv", index=False)
    best = cv.sort_values("cv_acc", ascending=False).groupby("family").head(1)
    print(f"{len(jobs)} settings tuned\n")
    print(best.round(4).to_string(index=False), "\n")

    # 2. Validation: fit each tuned model on the training split, score on validation.
    summary, val_conf = [], {}
    for _, r in best.iterrows():
        p = json.loads(r.params)
        X = design(struct, tokens, train, y, p.get("scaling", "z-score"), p.get("words", "all"))
        m = make_model(r.family, p).fit(X[train], y[train])
        pred = m.predict(X[val])
        f1 = per_sport_f1(y[val], pred)
        val_conf[r.family] = confusion_matrix(y[val], pred, labels=LABELS).tolist()
        summary.append(dict(family=r.family, params=r.params, cv_acc=r.cv_acc, cv_sd=r.cv_sd,
                            cv_se=r.cv_sd / np.sqrt(len(folds) - 1), train_acc=r.train_acc,
                            val_acc=np.mean(pred == y[val]),
                            **{f"val_f1_{k}": v for k, v in f1.items()},
                            val_macro_f1=np.mean(list(f1.values()))))
    S = pd.DataFrame(summary)
    S.to_csv(f"{OUT}/summary.csv", index=False)
    print(S.round(4).to_string(index=False), "\n")
    for fam, cm in val_conf.items():
        print(fam, "validation confusion (rows = true", LABELS, ")\n", np.array(cm))
    json.dump(val_conf, open(f"{OUT}/val_confusion.json", "w"))

    # Sensitivity curves for the report: best CV accuracy per value of each key setting.
    for fam, key in [("kNN", "k"), ("Random forest", "max_depth"), ("Naive Bayes", "alpha")]:
        sub = cv[cv.family == fam].copy()
        sub[key] = sub.params.map(lambda s: json.loads(s)[key])
        print(f"\n{fam}: best CV accuracy by {key}")
        print(sub.groupby(key, dropna=False)[["cv_acc", "train_acc"]].max().round(3).to_string())


def final_test(chosen="Naive Bayes"):
    """Retrain the chosen model on train + validation and score it once on test."""
    df = load_survey(DATA)
    df = df.loc[~blank_rows(df)].reset_index(drop=True)
    df["split"] = respondent_split(df["id"])
    y, ids = df["label"].to_numpy(), df["id"].to_numpy()
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}
    trval = np.where(df.split != "test")[0]
    test = np.where(df.split == "test")[0]
    S = pd.read_csv(f"{OUT}/summary.csv").set_index("family")
    p = json.loads(S.loc[chosen, "params"])

    # Stability on train + validation: grouped 5-fold CV repeated with 3 seeds.
    accs = []
    for seed in range(3):
        for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(trval, groups=ids[trval]):
            fit, held = trval[a], trval[b]
            X = design(struct, tokens, fit, y, p.get("scaling", "z-score"), p.get("words", "all"))
            m = make_model(chosen, p).fit(X[fit], y[fit])
            accs.append(np.mean(m.predict(X[held]) == y[held]))
    print(f"{chosen} on train+val, grouped 5-fold x 3: {100 * np.mean(accs):.1f}% "
          f"(SD {100 * np.std(accs):.1f}, min {100 * min(accs):.1f}, max {100 * max(accs):.1f})")

    X = design(struct, tokens, trval, y, p.get("scaling", "z-score"), p.get("words", "all"))
    m = make_model(chosen, p).fit(X[trval], y[trval])
    pred = m.predict(X[test])
    f1 = per_sport_f1(y[test], pred)
    cm = confusion_matrix(y[test], pred, labels=LABELS)
    result = dict(model=chosen, params=p, train_rows=len(trval), test_rows=len(test),
                  train_acc=float(np.mean(m.predict(X[trval]) == y[trval])),
                  test_acc=float(np.mean(pred == y[test])), test_f1=f1,
                  test_macro_f1=float(np.mean(list(f1.values()))), test_confusion=cm.tolist(),
                  repeated_cv=[float(a) for a in accs])
    json.dump(result, open(f"{OUT}/final_test.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "repeated_cv"}, indent=1))
    wrong = df.iloc[test][pred != y[test]]
    print(wrong.assign(pred=pred[pred != y[test]])[["label", "pred", "colour", "song", "food", "places", "seasons", "music"]].to_string())


if __name__ == "__main__":
    if not os.path.exists(f"{OUT}/summary.csv"):
        main()
    final_test()
