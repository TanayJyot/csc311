"""evaluation_checks.py - is the ~92% accuracy real, or is the evaluation leaking?

Checks, all on the chosen pipeline (Bernoulli naive Bayes, alpha 0.1, top 100 words)
unless stated otherwise:
 1. Label permutation: shuffle the labels and rerun grouped CV. A leak-free pipeline
    should fall to chance (~33%).
 2. Which questions carry the accuracy: CV with one question at a time.
 3. Explicit sport words in the free text (e.g. "soccer", "nba", "jays").
 4. Duplicate answers shared between different respondents.
 5. How lucky was our one test split: 200 other random 70/30 respondent splits.
 6. Selection bias: nested grouped CV (the whole naive Bayes grid tuned inside).
 7. Rows removed by cleaning: how the final model does on them.

Run: python evaluation_checks.py   (needs scikit-learn)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import BernoulliNB

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "methodology"))

from features import (LABELS, TEXT_COLS, clean_survey, design_matrix,  # noqa: E402
                      dev_test_split, fit_preprocessing, load_survey,
                      structured_features, tokenize)

DATA = str(ROOT / "csc311_project_training_202609.csv")
ALPHA, WORDS = 0.1, 100
rng = np.random.default_rng(0)

raw = load_survey(DATA)
df = clean_survey(raw)
df["split"] = dev_test_split(df["id"])
struct = structured_features(df).drop(columns=["price"])
tokens = {c: df[c].map(tokenize) for c in TEXT_COLS}
y, ids = df["label"].to_numpy(), df["id"].to_numpy()
dev = np.where(df.split == "dev")[0]
test = np.where(df.split == "test")[0]


def fit_predict(fit, held, labels=y, alpha=ALPHA, words=WORDS, cols=None, tok=None):
    """Fit preprocessing + NB on rows `fit` (with `labels`), predict rows `held`."""
    tok = tokens if tok is None else tok
    st = struct if cols is None else struct[cols]
    sub = {c: tok[c].iloc[fit].reset_index(drop=True) for c in TEXT_COLS}
    prep = fit_preprocessing(st.iloc[fit], sub, labels[fit], "z-score", words, 3)
    X = design_matrix(st, tok, prep)
    return BernoulliNB(alpha=alpha).fit(X[fit], labels[fit]).predict(X[held])


def grouped_cv(rows, labels=y, seed=0, **kw):
    accs = []
    for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(rows, groups=ids[rows]):
        fit, held = rows[a], rows[b]
        accs.append(np.mean(fit_predict(fit, held, labels, **kw) == labels[held]))
    return np.mean(accs)


print(f"reference: grouped 5-fold CV on dev = {100 * grouped_cv(dev):.1f}%, "
      f"test = {100 * np.mean(fit_predict(dev, test) == y[test]):.1f}%\n")

# 1. Permutation test ---------------------------------------------------------
perm = []
for _ in range(20):
    yp = y.copy()
    yp[dev] = rng.permutation(y[dev])
    perm.append(grouped_cv(dev, labels=yp))
print(f"1. shuffled labels, 20 runs: mean {100 * np.mean(perm):.1f}%, "
      f"max {100 * np.max(perm):.1f}%  (chance = 33.3%)\n")

# 2. One question at a time ---------------------------------------------------
empty = {c: tokens[c].map(lambda _: []) for c in TEXT_COLS}
groups = {"ease": ["ease"], "strategy": ["strategy"], "injury": ["injury"],
          "individual": ["individual"], "price": ["log_price"],
          "places": [c for c in struct if c.startswith("places=")],
          "seasons": [c for c in struct if c.startswith("seasons=")],
          "music": [c for c in struct if c.startswith("music=")]}
print("2. grouped CV accuracy using a single question:")
one = {}
for name, cols in groups.items():
    keep = cols + [c for c in ["ease", "log_price", "strategy", "injury", "individual"]
                   if c not in cols]                     # design_matrix needs these columns
    st = struct[keep].copy()
    for c in keep:
        if c not in cols:
            st[c] = 0.0
    accs = []
    for a, b in GroupKFold(5, shuffle=True, random_state=0).split(dev, groups=ids[dev]):
        fit, held = dev[a], dev[b]
        sub = {c: empty[c].iloc[fit].reset_index(drop=True) for c in TEXT_COLS}
        prep = fit_preprocessing(st.iloc[fit], sub, y[fit], "z-score", "all", 3)
        X = design_matrix(st, empty, prep)
        accs.append(np.mean(BernoulliNB(alpha=ALPHA).fit(X[fit], y[fit]).predict(X[held]) == y[held]))
    one[name] = np.mean(accs)
for c in TEXT_COLS:
    tok = {k: (tokens[k] if k == c else empty[k]) for k in TEXT_COLS}
    one[c] = grouped_cv(dev, tok=tok, words="all")
tok = empty
one["all structured (no text)"] = grouped_cv(dev, tok=empty, words="all")
one["all text (no structured)"] = grouped_cv(
    dev, tok=tokens, cols=["ease", "log_price", "strategy", "injury", "individual"])
for k, v in sorted(one.items(), key=lambda kv: -kv[1]):
    print(f"   {k:28s} {100 * v:5.1f}%")
print()

# 3. Explicit sport words -----------------------------------------------------
SPORT_WORDS = re.compile(
    r"\b(soccer|football|futbol|baseball|basketball|hoops?|nba|mlb|fifa|world cup|mls|"
    r"raptors|blue ?jays|jays|toronto fc|messi|ronaldo|lebron|jordan|kobe|curry|"
    r"home ?run|dunk|slam|goal|pitch|bat|court)\b", re.I)
text = df[TEXT_COLS].apply(lambda r: " | ".join(r), axis=1)
hit = text.str.contains(SPORT_WORDS).to_numpy()
pred_test = fit_predict(dev, test)
print(f"3. rows whose free text names the sport or an obvious cue: {hit.sum()} of {len(df)} "
      f"({100 * hit.mean():.0f}%)")
for name, mask in [("with cue", hit[test]), ("without cue", ~hit[test])]:
    print(f"   test accuracy {name:12s}: {100 * np.mean(pred_test[mask] == y[test][mask]):.1f}% "
          f"({mask.sum()} rows)")
print("   examples:", "; ".join(text[hit].head(5).str.slice(0, 70)), "\n")

# 4. Duplicates across respondents ----------------------------------------------
answer_cols = [c for c in df.columns if c not in ("id", "label", "split")]
key = df[answer_cols].apply(lambda r: "\x1f".join(x.strip().lower() for x in r), axis=1)
dup = key.duplicated(keep=False) & (df.groupby(key)["id"].transform("nunique") > 1)
cross = dup & df.groupby(key)["split"].transform("nunique").gt(1)
txt_key = df[TEXT_COLS].apply(lambda r: "\x1f".join(x.strip().lower() for x in r), axis=1)
txt_dup = (df.groupby(txt_key)["id"].transform("nunique") > 1) & (txt_key.str.len() > 10)
print(f"4. rows identical to another respondent's row: {dup.sum()} "
      f"(spanning dev and test: {cross.sum()}); "
      f"identical free text in all 4 questions: {txt_dup.sum()}\n")

# 5. Many other test splits ----------------------------------------------------
split_accs = []
for tr, te in GroupShuffleSplit(200, test_size=0.3, random_state=1).split(df, groups=ids):
    split_accs.append(np.mean(fit_predict(tr, te) == y[te]))
split_accs = np.array(split_accs)
ours = np.mean(pred_test == y[test])
print(f"5. 200 random 70/30 respondent splits: mean {100 * split_accs.mean():.1f}%, "
      f"SD {100 * split_accs.std():.1f}, 5th-95th pct "
      f"{100 * np.percentile(split_accs, 5):.1f}-{100 * np.percentile(split_accs, 95):.1f}%; "
      f"our split {100 * ours:.1f}% is at percentile {100 * np.mean(split_accs < ours):.0f}\n")

# 6. Nested CV ----------------------------------------------------------------
ALPHAS = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
WORD_OPTS = ["all", 25, 50, 100]
allrows = np.arange(len(df))
outer, chosen = [], []
for a, b in GroupKFold(5, shuffle=True, random_state=7).split(allrows, groups=ids):
    fit, held = allrows[a], allrows[b]
    best = max(((al, w) for al in ALPHAS for w in WORD_OPTS),
               key=lambda p: grouped_cv(fit, alpha=p[0], words=p[1], seed=1))
    chosen.append(best)
    outer.append(np.mean(fit_predict(fit, held, alpha=best[0], words=best[1]) == y[held]))
print(f"6. nested grouped CV on all 861 rows (NB grid tuned inside each outer fold): "
      f"{100 * np.mean(outer):.1f}% (folds {', '.join(f'{100 * x:.0f}' for x in outer)}); "
      f"settings chosen: {chosen}\n")

# 7. Rows removed by cleaning -----------------------------------------------------
kept = set(zip(df.id, df.label))
removed = raw[[(i, l) not in kept for i, l in zip(raw.id, raw.label)]].reset_index(drop=True)
both = pd.concat([df[raw.columns], removed], ignore_index=True)
s2 = structured_features(both).drop(columns=["price"])
t2 = {c: both[c].map(tokenize) for c in TEXT_COLS}
sub = {c: t2[c].iloc[dev].reset_index(drop=True) for c in TEXT_COLS}
prep = fit_preprocessing(s2.iloc[dev], sub, y[dev], "z-score", WORDS, 3)
X2 = design_matrix(s2, t2, prep)
r = np.arange(len(df), len(both))
pr = BernoulliNB(alpha=ALPHA).fit(X2[dev], y[dev]).predict(X2[r])
print(f"7. {len(removed)} rows removed by cleaning; model gets {np.sum(pr == removed.label)} right "
      f"({100 * np.mean(pr == removed.label):.0f}%). Blank rows can only be guessed.")
