"""Scaling experiments: synthetic-only, real-only and real+synthetic training, tested on real rows."""
import glob, json, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common
from features import structured_features, tokenize, TEXT_COLS, FEATURE_COLS
from models import FAMILIES
warnings.filterwarnings("ignore")

MODELS = {"Random forest": ("RandomForest", {"n": 500, "f": "log2", "d": None, "l": 1}, {"min_df": 1}),
          "Bernoulli NB": ("BernoulliNB", {"a": 0.5}, {"min_df": 1}),
          "Softmax regression": ("SoftmaxRegression", {"C": 0.5, "pen": "l1"}, {"min_df": 2})}

def load_synth():
    rows, bad = [], 0
    for f in sorted([str(__import__("pathlib").Path(__file__).resolve().parent / "synthetic_respondents.jsonl")]):
        for line in open(f, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1; continue
            r["src"] = f.split("/")[-1][:7]
            rows.append(r)
    S = pd.DataFrame(rows)
    for c in FEATURE_COLS + ["label"]:
        S[c] = S.get(c, "").fillna("").astype(str) if c in S else ""
    S["id"] = "s" + S["id"].astype(str)
    S = S[S.label.isin(["Soccer", "Baseball", "Basketball"])]
    ok = S.groupby("id").label.agg(lambda s: len(s) == 3 and s.nunique() == 3)
    S = S[S.id.isin(ok[ok].index)].reset_index(drop=True)
    print(f"synthetic: {len(S)} rows, {S.id.nunique()} respondents, {bad} unparseable lines", flush=True)
    return S

real, _, _ = common.load_all()
synth = load_synth()
cols = ["id", "label"] + FEATURE_COLS
ALL = pd.concat([real[cols].assign(kind="real", split=real["split"]),
                 synth[cols].assign(kind="synth", split="synth")], ignore_index=True)
STRUCT = structured_features(ALL).drop(columns=["price"])
TOK = {c: ALL[c].map(tokenize) for c in TEXT_COLS}
y = ALL.label.to_numpy(); ids = ALL.id.to_numpy()
real_all = np.where(ALL.kind == "real")[0]
real_test = np.where(ALL.split == "test")[0]
real_dev = np.where((ALL.kind == "real") & (ALL.split != "test"))[0]
synth_idx = np.where(ALL.kind == "synth")[0]

def rows_of(resp_ids):
    s = set(resp_ids); return np.where([i in s for i in ids])[0]

def evaluate(name, train, tests):
    fam, p, feat = MODELS[name]
    X = common.design(STRUCT, TOK, train, **feat)
    m = FAMILIES[fam][0](p).fit(X[train], y[train])
    out = {}
    for tname, te in tests.items():
        P = common.scores_to_proba(m, X[te]); c = m.classes_
        out[tname] = (c[P.argmax(1)] == y[te]).mean()
        out[tname + "_joint"] = (c[common.joint_assign(P, ids[te])] == y[te]).mean()
    return out

def job(exp, name, n_real, n_syn, seed):
    rng = np.random.default_rng(seed)
    parts = []
    if n_real:
        parts.append(rows_of(rng.choice(pd.unique(ids[real_dev]), n_real, replace=False)))
    if n_syn:
        parts.append(rows_of(rng.choice(pd.unique(ids[synth_idx]), n_syn, replace=False)))
    train = np.concatenate(parts)
    tests = {"real_test": real_test}
    if not n_real:
        tests["real_all"] = real_all
    return dict(exp=exp, model=name, n_real=n_real, n_syn=n_syn, seed=seed, **evaluate(name, train, tests))

NS = pd.unique(ids[synth_idx]).size; NR = pd.unique(ids[real_dev]).size
syn_sizes = [s for s in [10, 20, 40, 80, 160, 320, 640, 1000, 1200] if s < NS] + [NS]
real_sizes = [s for s in [10, 20, 40, 80, 160] if s < NR] + [NR]
mix_sizes = sorted({s for s in [0, 50, 100, 200, 400, 800, NS] if s <= NS})
jobs = []
for name in MODELS:
    for seed in range(5):
        jobs += [("synthetic only", name, 0, n, seed) for n in syn_sizes]
        jobs += [("real only", name, n, 0, seed) for n in real_sizes]
        jobs += [("real + synthetic", name, NR, n, seed) for n in mix_sizes if n > 0]
R = pd.DataFrame(Parallel(n_jobs=4)(delayed(job)(*j) for j in jobs))
R.to_csv("scaling_results.csv", index=False)

pd.set_option("display.width", 200)
agg = lambda d, keys, cols: d.groupby(keys)[cols].agg(["mean", "std"]).round(3)
print("\n== synthetic only (train n synthetic respondents; test = real) ==")
print(agg(R[R.exp == "synthetic only"], ["model", "n_syn"], ["real_all", "real_all_joint", "real_test", "real_test_joint"]).to_string())
print("\n== real only (train n real dev respondents; test = real test) ==")
print(agg(R[R.exp == "real only"], ["model", "n_real"], ["real_test", "real_test_joint"]).to_string())
print("\n== all real dev + n synthetic (test = real test) ==")
print(agg(R[R.exp == "real + synthetic"], ["model", "n_syn"], ["real_test", "real_test_joint"]).to_string())

def fit(n, acc):
    f = lambda n, a, b, c: c + a * np.power(n, -b)    # error = c + a n^-b
    try:
        (a, b, c), _ = curve_fit(f, n, 1 - acc, p0=[0.5, 0.5, 0.05], bounds=([0, 0, 0], [10, 3, 1]), maxfev=20000)
        return dict(a=a, b=b, irreducible_err=c, asymptote_acc=1 - c)
    except Exception as e:
        return dict(err=str(e))
print("\n== power-law fits, error = c + a * n^-b (n = respondents) ==")
for name in MODELS:
    for exp, ncol in [("synthetic only", "n_syn"), ("real only", "n_real")]:
        d = R[(R.exp == exp) & (R.model == name)].groupby(ncol).real_test.mean()
        print(name, "|", exp, "|", {k: round(v, 3) for k, v in fit(d.index.to_numpy(float), d.to_numpy()).items()})
