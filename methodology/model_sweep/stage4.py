"""Stage 4: (a) soft-voting ensemble in the same repeated grouped CV as stage 2;
(b) one-time evaluation on the instructors' held-out test respondents."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import common
from models import FAMILIES
warnings.filterwarnings("ignore")
df, struct, tokens = common.load_all()
y = df["label"].to_numpy(); ids = df["id"].to_numpy()
dev = np.where(df.split != "test")[0]; test = np.where(df.split == "test")[0]
C = json.load(open("candidates.json"))
TOP3 = C[:3]

def fit_proba(cand, fit, held):
    fam, p, feat = cand
    X = common.design(struct, tokens, fit, **feat)
    m = FAMILIES[fam][0](p).fit(X[fit], y[fit])
    P = common.scores_to_proba(m, X[held])
    return P, m.classes_

folds = [(dev[a], dev[b]) for rep in range(3)
         for a, b in GroupKFold(5, shuffle=True, random_state=10 + rep).split(dev, groups=ids[dev])]
rows = []
for f, h in folds:
    Ps = [fit_proba(c, f, h) for c in TOP3]; cls = Ps[0][1]
    for name, P in [(c[0], P) for c, (P, _) in zip(TOP3, Ps)] + [("SoftVote(top3)", np.mean([P for P, _ in Ps], 0))]:
        rows.append(dict(model=name, acc=(cls[P.argmax(1)] == y[h]).mean(),
                         joint=(cls[common.joint_assign(P, ids[h])] == y[h]).mean()))
cv = pd.DataFrame(rows).groupby("model").agg(cv_acc=("acc", "mean"), cv_sd=("acc", "std"),
                                             cv_joint=("joint", "mean"), cv_joint_sd=("joint", "std"))
print("== repeated grouped CV on development set (15 folds) ==\n", cv.round(3).to_string())

out = []
Ps = []
for c in C:
    P, cls = fit_proba(c, dev, test); Ps.append(P)
    out.append(dict(model=c[0], test_acc=(cls[P.argmax(1)] == y[test]).mean(),
                    test_joint=(cls[common.joint_assign(P, ids[test])] == y[test]).mean()))
P = np.mean(Ps[:3], 0)
out.append(dict(model="SoftVote(top3)", test_acc=(cls[P.argmax(1)] == y[test]).mean(),
                test_joint=(cls[common.joint_assign(P, ids[test])] == y[test]).mean()))
print(f"== held-out test ({len(test)} rows, {len(set(ids[test]))} respondents), trained on dev ==")
print(pd.DataFrame(out).round(3).to_string(index=False))
