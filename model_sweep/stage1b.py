"""Stage 1: screen every family on the development set (instructors' train+val),
grouped 5-fold CV x 2 repeats. The test respondents are never touched."""
import sys, time, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
import common
from models_extra import EXTRA as FAMILIES
warnings.filterwarnings("ignore")

df, struct, tokens = common.load_all()
dev = np.where(df["split"] != "test")[0]
y_all = df["label"].to_numpy(); ids_all = df["id"].to_numpy()
FEAT = dict(min_df=2, text="binary")

folds = []
for rep in range(2):
    for a, b in GroupKFold(5, shuffle=True, random_state=rep).split(dev, groups=ids_all[dev]):
        folds.append((dev[a], dev[b]))
X_by_fold = [common.design(struct, tokens, f, **FEAT) for f, _ in folds]

def run(fam, p):
    make = FAMILIES[fam][0]
    acc, jacc = [], []
    t0 = time.time()
    for X, (f, h) in zip(X_by_fold, folds):
        m = make(p).fit(X[f], y_all[f])
        P = common.scores_to_proba(m, X[h]); cls = m.classes_
        acc.append((cls[P.argmax(1)] == y_all[h]).mean())
        jacc.append((cls[common.joint_assign(P, ids_all[h])] == y_all[h]).mean())
    return dict(family=fam, params=json.dumps(p, default=str), acc=np.mean(acc), sd=np.std(acc),
                joint=np.mean(jacc), secs=time.time() - t0)

jobs = [(fam, p) for fam, (_, g) in FAMILIES.items() for p in g]
print(len(jobs), "configs x", len(folds), "folds", flush=True)
res = Parallel(n_jobs=2, verbose=0)(delayed(run)(f, p) for f, p in jobs)
R = pd.DataFrame(res).sort_values("acc", ascending=False)
R.to_csv("stage1b_all.csv", index=False)
best = R.groupby("family").head(1).reset_index(drop=True)
best.to_csv("stage1b_best_per_family.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(best[["family", "acc", "sd", "joint", "params"]].round(3).to_string())
