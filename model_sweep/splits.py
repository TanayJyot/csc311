"""Stage 3: does the train/val/test ratio matter? For each scheme and 30 random seeds:
split respondents, pick the best candidate on validation, refit on train+val, score on test."""
import json, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import common
from models import FAMILIES
warnings.filterwarnings("ignore")

df, struct, tokens = common.load_all()
y = df["label"].to_numpy(); ids = df["id"].to_numpy()
CANDS = json.load(open("candidates.json"))   # list of [family, params, featdict]
SCHEMES = {"50/20/30 (instructors)": (0.5, 0.2, 0.3), "70/15/15": (0.7, 0.15, 0.15),
           "60/20/20": (0.6, 0.2, 0.2), "80/10/10": (0.8, 0.1, 0.1)}

def one(name, fr, seed):
    u = np.random.default_rng(seed).permutation(pd.unique(ids))
    nt, nv = int(len(u) * fr[2]), int(len(u) * fr[1])
    te_ids, va_ids = set(u[:nt]), set(u[nt:nt + nv])
    te = np.where([i in te_ids for i in ids])[0]; va = np.where([i in va_ids for i in ids])[0]
    tr = np.setdiff1d(np.arange(len(y)), np.r_[te, va])
    vals = []
    for fam, p, feat in CANDS:
        X = common.design(struct, tokens, tr, **feat)
        m = FAMILIES[fam][0](p).fit(X[tr], y[tr])
        vals.append((m.predict(X[va]) == y[va]).mean())
    k = int(np.argmax(vals)); fam, p, feat = CANDS[k]
    trva = np.r_[tr, va]
    X = common.design(struct, tokens, trva, **feat)
    m = FAMILIES[fam][0](p).fit(X[trva], y[trva])
    P = common.scores_to_proba(m, X[te])
    test = (m.classes_[P.argmax(1)] == y[te]).mean()
    jt = (m.classes_[common.joint_assign(P, ids[te])] == y[te]).mean()
    # oracle: what the best candidate on this test set would have scored
    return dict(scheme=name, seed=seed, chosen=fam, val=vals[k], test=test, joint_test=jt)

rows = Parallel(n_jobs=2)(delayed(one)(n, fr, s) for n, fr in SCHEMES.items() for s in range(30))
R = pd.DataFrame(rows); R.to_csv("stage3_splits.csv", index=False)
R["gap"] = (R.val - R.test).abs()
S = R.groupby("scheme").agg(test_mean=("test", "mean"), test_sd=("test", "std"),
                            joint_mean=("joint_test", "mean"), val_test_gap=("gap", "mean"),
                            n_test_rows_hint=("seed", "size"))
print(S.round(3).to_string())
print(R.groupby("scheme").chosen.value_counts().unstack(fill_value=0).to_string())
