"""model_selection.py - the model selection reported in the final report.

Protocol (Methodology section):
0. Clean first: drop the 7 blank rows and the respondents left without one row per
   sport (861 rows, 287 respondents). Then split respondents 70/30 with the
   instructors' split_by_student shuffle (seed 42): development and test.
1. Tune kNN, random forest and Bernoulli naive Bayes with grouped 5-fold CV on it;
   preprocessing is fit on the four training folds each time.
2. Choose the final model from the CV results: accuracy, per-sport F1 and the
   confusion matrix of the held-out (out-of-fold) predictions. Models within one
   standard error of the best are tied, and the simplest of the tied models wins.
3. Retrain the chosen model on all 603 development rows and score it once on the
   test split.

Run: python model_selection.py   (needs scikit-learn; writes model_selection_results/)
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "methodology"))   # features.py

from features import (LABELS, TEXT_COLS, clean_survey, design_matrix,  # noqa: E402
                      dev_test_split, fit_preprocessing, load_survey,
                      structured_features, tokenize)

DATA = str(ROOT / "csc311_project_training_202609.csv")
OUT = str(HERE / "model_selection_results")
MIN_DF = 3                      # keep words used by at least 3 training rows
SIMPLEST_FIRST = ["Naive Bayes", "kNN", "Random forest"]   # tie-break order for pred.py


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


def load_data():
    """Cleaned survey (one row per sport for every respondent) split 70/30 by respondent."""
    raw = load_survey(DATA)
    df = clean_survey(raw)
    print(f"cleaned: {len(raw)} rows / {raw['id'].nunique()} respondents -> "
          f"{len(df)} rows / {df['id'].nunique()} respondents")
    df["split"] = dev_test_split(df["id"])
    print(df.groupby("split").agg(respondents=("id", "nunique"), rows=("id", "size")), "\n")
    struct = structured_features(df).drop(columns=["price"])
    tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}
    return df, struct, tokens


def main():
    os.makedirs(OUT, exist_ok=True)
    df, struct, tokens = load_data()
    y, ids = df["label"].to_numpy(), df["id"].to_numpy()
    dev = np.where(df.split == "dev")[0]                   # 70% development set
    test = np.where(df.split == "test")[0]
    folds = [(dev[a], dev[b]) for a, b in
             GroupKFold(5, shuffle=True, random_state=0).split(dev, groups=ids[dev])]

    # 1. Grouped 5-fold CV on the development set. Design matrices are cached per
    #    (fold, scaling, words) because they do not depend on the model.
    cache = {}

    def X_for(i, scaling, words):
        if (i, scaling, words) not in cache:
            cache[(i, scaling, words)] = design(struct, tokens, folds[i][0], y, scaling, words)
        return cache[(i, scaling, words)]

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
    print(f"{len(jobs)} settings tuned with grouped 5-fold CV on {len(dev)} rows\n")

    # 2. Held-out (out-of-fold) predictions of each family's best setting:
    #    per-sport F1 and confusion matrix, all from the same five folds.
    summary, cv_conf = [], {}
    for _, r in best.iterrows():
        p = json.loads(r.params)
        pred = np.empty(len(y), dtype=object)
        for i, (fit, held) in enumerate(folds):
            X = X_for(i, p.get("scaling", "z-score"), p.get("words", "all"))
            pred[held] = make_model(r.family, p).fit(X[fit], y[fit]).predict(X[held])
        f1 = per_sport_f1(y[dev], pred[dev])
        cv_conf[r.family] = confusion_matrix(y[dev], pred[dev], labels=LABELS).tolist()
        summary.append(dict(family=r.family, params=r.params, cv_acc=r.cv_acc, cv_sd=r.cv_sd,
                            cv_se=r.cv_sd / np.sqrt(len(folds) - 1), train_acc=r.train_acc,
                            **{f"cv_f1_{k}": v for k, v in f1.items()},
                            cv_macro_f1=np.mean(list(f1.values()))))
    S = pd.DataFrame(summary).sort_values("cv_acc", ascending=False)
    top = S.iloc[0]
    tied = S[S.cv_acc >= top.cv_acc - top.cv_se].family.tolist()
    chosen = min(tied, key=SIMPLEST_FIRST.index)
    S["tied_with_best"] = S.family.isin(tied)
    S["chosen"] = S.family == chosen
    S.to_csv(f"{OUT}/summary.csv", index=False)
    json.dump(cv_conf, open(f"{OUT}/cv_confusion.json", "w"))
    print(S.round(4).to_string(index=False), "\n")
    for fam, cm in cv_conf.items():
        print(fam, "CV confusion (rows = true", LABELS, ")\n", np.array(cm))
    print(f"\nbest CV accuracy: {top.family}; within one SE ({100 * top.cv_se:.1f} points): {tied}; "
          f"chosen (simplest of the tied): {chosen}")

    # Sensitivity: best CV accuracy per value of each key setting.
    for fam, key in [("kNN", "k"), ("kNN", "scaling"), ("Random forest", "max_depth"),
                     ("Random forest", "max_features"), ("Naive Bayes", "alpha"),
                     ("Naive Bayes", "words")]:
        sub = cv[cv.family == fam].copy()
        sub[key] = sub.params.map(lambda s: str(json.loads(s)[key]))
        print(f"\n{fam}: best CV accuracy by {key}")
        print(sub.groupby(key)[["cv_acc", "train_acc"]].max().round(3).to_string())

    # 3. Stability of the chosen setting: grouped 5-fold CV repeated with 3 seeds.
    p = json.loads(S.loc[S.chosen, "params"].iloc[0])
    accs = []
    for seed in range(3):
        for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(dev, groups=ids[dev]):
            fit, held = dev[a], dev[b]
            X = design(struct, tokens, fit, y, p.get("scaling", "z-score"), p.get("words", "all"))
            accs.append(np.mean(make_model(chosen, p).fit(X[fit], y[fit]).predict(X[held]) == y[held]))
    print(f"\n{chosen}, grouped 5-fold CV x 3 seeds: {100 * np.mean(accs):.1f}% "
          f"(SD {100 * np.std(accs):.1f}, min {100 * min(accs):.1f}, max {100 * max(accs):.1f})")

    # 4. Retrain on all development rows and score the test split once.
    X = design(struct, tokens, dev, y, p.get("scaling", "z-score"), p.get("words", "all"))
    m = make_model(chosen, p).fit(X[dev], y[dev])
    pred = m.predict(X[test])
    f1 = per_sport_f1(y[test], pred)
    result = dict(model=chosen, params=p, train_rows=len(dev), test_rows=len(test),
                  train_acc=float(np.mean(m.predict(X[dev]) == y[dev])),
                  test_acc=float(np.mean(pred == y[test])), test_f1=f1,
                  test_macro_f1=float(np.mean(list(f1.values()))),
                  test_confusion=confusion_matrix(y[test], pred, labels=LABELS).tolist(),
                  repeated_cv=[float(a) for a in accs])
    json.dump(result, open(f"{OUT}/final_test.json", "w"), indent=1)
    print("\nfinal test:", json.dumps({k: v for k, v in result.items() if k != "repeated_cv"}, indent=1))
    wrong = df.iloc[test][pred != y[test]].assign(pred=pred[pred != y[test]])
    print(wrong[["label", "pred", "colour", "song", "food", "places", "seasons", "music"]].to_string())


if __name__ == "__main__":
    main()
