"""PCA + SVM, evaluated exactly like stage 2 / stage 4.
Development set only: grouped 5-fold CV x 3 repeats (15 folds), PCA fit on the
training folds only. Then the best settings are refit on dev and scored once on
the held-out test respondents. SVMs without PCA are run on the same folds as a baseline."""
import json, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC, LinearSVC
import common
from models import grid
warnings.filterwarnings("ignore")

df, struct, tokens = common.load_all()
y = df["label"].to_numpy(); ids = df["id"].to_numpy()
dev = np.where(df.split != "test")[0]; test = np.where(df.split == "test")[0]
FEATS = {"bin1": dict(min_df=1), "bin3": dict(min_df=3)}
KS = [5, 10, 20, 40, 80, 120, 200, "all"]   # "all" = every component (a rotation, no reduction)
SVMS = ([("linear", p) for p in grid(C=[0.003, 0.01, 0.03, 0.1, 0.3, 1])] +
        [("rbf", p) for p in grid(C=[1, 3, 10, 30], g=["scale", 0.001, 0.003, 0.01])])

def make_svm(kind, p):
    if kind == "linear":
        return LinearSVC(C=p["C"], max_iter=50000)
    return SVC(C=p["C"], gamma=p["g"], kernel="rbf")

def project(X, fit, held, k, whiten):
    """PCA fit on rows `fit`; components are nested, so one full fit serves every k."""
    pca = PCA(svd_solver="full").fit(X[fit])
    k = pca.n_components_ if k == "all" else min(k, pca.n_components_)
    W = pca.components_[:k].T
    if whiten:
        W = W / np.sqrt(pca.explained_variance_[:k])
    mu = pca.mean_
    return (X[fit] - mu) @ W, (X[held] - mu) @ W, pca.explained_variance_ratio_[:k].sum()

def evaluate(fit, held, fk, k, whiten, kind, p):
    X = common.design(struct, tokens, fit, **FEATS[fk])
    if k is None:
        Zf, Zh, ev = X[fit], X[held], 1.0
    else:
        Zf, Zh, ev = project(X, fit, held, k, whiten)
    m = make_svm(kind, p).fit(Zf, y[fit])
    P = common.scores_to_proba(m, Zh)
    return ((m.classes_[P.argmax(1)] == y[held]).mean(),
            (m.classes_[common.joint_assign(P, ids[held])] == y[held]).mean(), ev)

folds = [(dev[a], dev[b]) for rep in range(3)
         for a, b in GroupKFold(5, shuffle=True, random_state=10 + rep).split(dev, groups=ids[dev])]

# Cache the projections per fold, then fit every SVM on them.
def fold_jobs(fi, f, h):
    out = []
    for fk in FEATS:
        X = common.design(struct, tokens, f, **FEATS[fk])
        pca = PCA(svd_solver="full").fit(X[f])
        for k in [None] + KS:
            for wh in ([False] if k is None else [False, True]):
                if k is None:
                    Zf, Zh, ev, kk = X[f], X[h], 1.0, X.shape[1]
                else:
                    kk = pca.n_components_ if k == "all" else min(k, pca.n_components_)
                    W = pca.components_[:kk].T
                    if wh:
                        W = W / np.sqrt(pca.explained_variance_[:kk])
                    Zf, Zh = (X[f] - pca.mean_) @ W, (X[h] - pca.mean_) @ W
                    ev = pca.explained_variance_ratio_[:kk].sum()
                for kind, p in SVMS:
                    m = make_svm(kind, p).fit(Zf, y[f])
                    P = common.scores_to_proba(m, Zh); c = m.classes_
                    out.append(dict(fold=fi, feat=fk, k="none" if k is None else str(k), dims=kk,
                                    whiten=wh, kernel=kind, params=json.dumps(p),
                                    var_explained=ev, acc=(c[P.argmax(1)] == y[h]).mean(),
                                    joint=(c[common.joint_assign(P, ids[h])] == y[h]).mean()))
    return out

t0 = time.time()
rows = [r for chunk in Parallel(n_jobs=4)(delayed(fold_jobs)(i, f, h) for i, (f, h) in enumerate(folds))
        for r in chunk]
print(f"{len(rows)} fits in {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
keys = ["feat", "k", "whiten", "kernel", "params"]
S = (R.groupby(keys).agg(dims=("dims", "mean"), var_explained=("var_explained", "mean"),
                         acc=("acc", "mean"), acc_sd=("acc", "std"),
                         joint=("joint", "mean"), joint_sd=("joint", "std")).reset_index()
       .sort_values("acc", ascending=False))
S.to_csv("pca_svm_all.csv", index=False)
pd.set_option("display.width", 220)

print("\n== top 15 settings (15-fold grouped CV on dev) ==")
print(S.head(15).round(3).to_string(index=False))
print("\n== best per kernel x PCA / no PCA ==")
S["pca"] = np.where(S.k == "none", "no PCA", "PCA")
best = S.groupby(["kernel", "pca"]).head(1).sort_values("acc", ascending=False)
print(best.round(3).to_string(index=False))
print("\n== best CV accuracy by number of components (any kernel, whiten, C) ==")
print(S.groupby(["feat", "k"]).agg(dims=("dims", "first"), var_explained=("var_explained", "first"),
                           best_acc=("acc", "max")).sort_values(["feat", "dims"]).round(3).to_string())
print("\n== whitening effect (best per k x kernel) ==")
print(S[S.k != "none"].groupby(["kernel", "whiten"]).acc.max().unstack().round(3).to_string())

# Held-out test: refit each kernel/PCA winner on all dev rows, score once.
out = []
for _, b in best.iterrows():
    k = None if b.k == "none" else (b.k if b.k == "all" else int(b.k))
    a, j, ev = evaluate(dev, test, b.feat, k, bool(b.whiten), b.kernel, json.loads(b.params))
    out.append(dict(model=f"{'PCA+' if k is not None else ''}{b.kernel} SVM", feat=b.feat, k=b.k,
                    whiten=b.whiten, params=b.params, cv_acc=b.acc, test_acc=a, test_joint=j))
print(f"\n== held-out test ({len(test)} rows, {len(set(ids[test]))} respondents), trained on dev ==")
T = pd.DataFrame(out); print(T.round(3).to_string(index=False))
T.to_csv("pca_svm_test.csv", index=False)
