"""Stage 2: wider hyperparameter + feature search for the top families.
Development set only, grouped 5-fold CV x 3 repeats (15 fits per config)."""
import json, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import common
from models import FAMILIES, grid
warnings.filterwarnings("ignore")

df, struct, tokens = common.load_all()
dev = np.where(df["split"] != "test")[0]
y = df["label"].to_numpy(); ids = df["id"].to_numpy()
FEATS = {"bin2": dict(min_df=2), "bin1": dict(min_df=1), "bin3": dict(min_df=3),
         "bin5": dict(min_df=5), "tfidf2": dict(min_df=2, text="tfidf"), "notext": dict(use_text=False)}
folds = [(dev[a], dev[b]) for rep in range(3)
         for a, b in GroupKFold(5, shuffle=True, random_state=10 + rep).split(dev, groups=ids[dev])]
XS = {fk: [common.design(struct, tokens, f, **fv) for f, _ in folds] for fk, fv in FEATS.items()}

G = {
 "GradientBoosting": grid(n=[100, 200, 400], lr=[0.03, 0.1], d=[2, 3, 4]),
 "BernoulliNB": grid(a=[0.5, 1, 2, 3, 5, 8, 12]),
 "RandomForest": grid(n=[500], f=["sqrt", "log2", 0.05, 0.2], d=[None, 8, 15], l=[1, 2]),
 "LDA": grid(s=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, "auto"]),
 "LeastSquares(Ridge)": grid(a=[1, 3, 5, 10, 20, 40, 80]),
 "LinearSVM": grid(C=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
 "SoftmaxRegression": grid(C=[0.05, 0.1, 0.2, 0.3, 0.5, 1, 2], pen=["l1", "l2"]),
 "AdaBoost": grid(d=[1, 2, 3], n=[200, 500], lr=[0.05, 0.1, 0.3]),
}

def run(fam, p, fk):
    accs, jaccs, lls = [], [], []
    for X, (f, h) in zip(XS[fk], folds):
        m = FAMILIES[fam][0](p).fit(X[f], y[f])
        P = common.scores_to_proba(m, X[h]); c = m.classes_
        accs.append((c[P.argmax(1)] == y[h]).mean())
        jaccs.append((c[common.joint_assign(P, ids[h])] == y[h]).mean())
        lls.append(log_loss(y[h], np.clip(P, 1e-6, 1), labels=c))
    return dict(family=fam, feat=fk, params=json.dumps(p, default=str), acc=np.mean(accs),
                acc_sd=np.std(accs), joint=np.mean(jaccs), joint_sd=np.std(jaccs), logloss=np.mean(lls))

jobs = [(fam, p, fk) for fam, g in G.items() for p in g for fk in FEATS]
print(len(jobs), "configs", flush=True); t0 = time.time()
R = pd.DataFrame(Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs))
R.to_csv("stage2_all.csv", index=False)
print(f"{time.time()-t0:.0f}s")
pd.set_option("display.width", 220, "display.max_colwidth", 60)
cols = ["family", "feat", "acc", "acc_sd", "joint", "joint_sd", "logloss", "params"]
print("== best per family by plain accuracy ==")
print(R.sort_values("acc", ascending=False).groupby("family").head(1)[cols].round(3).to_string(index=False))
print("== best per family by joint-assignment accuracy ==")
print(R.sort_values(["joint", "logloss"], ascending=[False, True]).groupby("family").head(1)[cols].round(3).to_string(index=False))
print("== feature variant effect (mean of each family's best) ==")
print(R.groupby(["family", "feat"]).acc.max().unstack().round(3).to_string())
