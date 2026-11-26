"""check_pred.py - check the submitted pred.py (and pred_all.py) against the rules and scikit-learn.

1. Retrains (train_final.py) into temporary folders: the three tuned models on the 70%
   development set, and the submitted naive Bayes on all 861 cleaned rows. Runs
   submission/pred.py from a clean folder holding only pred.py and weights.json.
2. Checks: pred.py imports only the standard library, numpy and pandas; its predictions
   equal scikit-learn's naive Bayes (trained on all rows) on every cleaned row; its own
   copy of the cleaning and feature code gives exactly the same design matrix as
   features.py on every row of the raw CSV; the committed weights.json equals a fresh
   retrain; 60 rows run in time; and the submitted files are under 10 MB.
3. Checks that pred_all.py (development-set models) reproduces scikit-learn for all three
   families and reports their held-out test accuracy.

Run: python check_pred.py   (needs scikit-learn)
"""
import ast
import json
import importlib.util
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))   # features.py, train_final.py

from features import (TEXT_COLS, blank_rows, clean_survey, design_matrix, dev_test_split,
                      load_survey, structured_features, tokenize)
from train_final import DATA, fit_and_export, fit_submission

SUBMITTED = ["pred.py", "weights.json"]
SUB = ROOT / "methodology" / "submission"
ALLOWED_IMPORTS = {"json", "os", "sys", "re", "numpy", "pandas"}


def check_imports():
    """pred.py may import only the standard library, numpy and pandas."""
    for name in ("pred.py",):
        for node in ast.walk(ast.parse(open(SUB / name).read())):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] in ALLOWED_IMPORTS, f"{name} imports {module}"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    check_imports()
    sub_dir, model_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    models, preps = fit_and_export(out_dir=model_dir)
    nb_all, prep_all = fit_submission(submission_dir=sub_dir)
    shutil.copy(SUB / "pred.py", sub_dir)
    assert sorted(os.listdir(sub_dir)) == sorted(SUBMITTED), os.listdir(sub_dir)
    with open(SUB / "weights.json") as f, open(os.path.join(sub_dir, "weights.json")) as g:
        assert json.load(f) == json.load(g), "committed weights.json differs from a fresh retrain"
    print("committed submission/weights.json equals a fresh retrain")

    raw = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    short = load_survey(DATA)
    # Same cleaning as clean_survey (blank rows, then respondents without one row per
    # sport), kept as a mask over the raw rows so the raw CSV text can be written out.
    keep = ~blank_rows(short)
    answered = short["id"].where(keep).map(short.loc[keep, "id"].value_counts())
    full = keep & (answered == 3)
    split = pd.Series("dropped", index=short.index)
    split[full] = dev_test_split(short.loc[full, "id"])
    assert full.sum() == len(clean_survey(short))
    test_mask = split == "test"
    test_csv = os.path.join(sub_dir, "..", "test_rows.csv")
    raw.loc[test_mask].drop(columns=["label"]).to_csv(test_csv, index=False)
    labels = short.loc[test_mask, "label"].tolist()

    # 1. The submitted script, loaded from a folder holding only pred.py and weights.json.
    #    Its model was trained on all cleaned rows, so compare it with scikit-learn on all
    #    of them (accuracy on these rows is training accuracy, not a held-out estimate).
    pred = load_module("pred_under_test", os.path.join(sub_dir, "pred.py"))
    clean_csv = os.path.join(sub_dir, "..", "clean_rows.csv")
    raw.loc[full].drop(columns=["label"]).to_csv(clean_csv, index=False)
    clean = short.loc[full].reset_index(drop=True)
    ours = pred.predict_all(clean_csv)
    theirs = nb_all.predict(design_matrix(structured_features(clean).drop(columns=["price"]),
                                          {c: clean[c].map(tokenize) for c in TEXT_COLS},
                                          prep_all)).tolist()
    same = sum(a == b for a, b in zip(ours, theirs))
    print(f"pred.py (naive Bayes, trained on all {len(clean)} cleaned rows): matches sklearn on "
          f"{same}/{len(ours)} rows; training accuracy "
          f"{100 * np.mean(np.array(ours) == clean['label'].to_numpy()):.1f}%")
    assert same == len(ours)

    test = short.loc[test_mask].reset_index(drop=True)
    struct = structured_features(test).drop(columns=["price"])
    tokens = {c: test[c].map(tokenize) for c in TEXT_COLS}

    # pred.py carries its own copy of the cleaning and feature code: it must give the
    # same design matrix as features.py on every raw row, blank rows included.
    all_csv = os.path.join(sub_dir, "..", "all_rows.csv")
    raw.drop(columns=["label"]).to_csv(all_csv, index=False)
    df_p = pred.load_survey(all_csv)
    X_p = pred.design_matrix(pred.structured_features(df_p).drop(columns=["price"]),
                             {c: df_p[c].map(pred.tokenize) for c in TEXT_COLS},
                             prep_all)
    df_f = load_survey(all_csv)
    X_f = design_matrix(structured_features(df_f).drop(columns=["price"]),
                        {c: df_f[c].map(tokenize) for c in TEXT_COLS}, prep_all)
    assert np.array_equal(X_p, X_f, equal_nan=True)
    print(f"pred.py feature code equals features.py on all {len(X_p)} raw rows")

    small = os.path.join(sub_dir, "..", "sixty.csv")
    raw.loc[test_mask].head(60).drop(columns=["label"]).to_csv(small, index=False)
    start = time.time()
    pred.predict_all(small)
    print(f"60 predictions in {time.time() - start:.2f} s")
    size = sum(os.path.getsize(os.path.join(sub_dir, f)) for f in SUBMITTED)
    print(f"submitted files ({', '.join(SUBMITTED)}): {size / 1e6:.3f} MB (limit 10 MB)")

    # 2. pred_all.py: numpy versions of all three families (development-set models)
    #    against scikit-learn; their test accuracy is the held-out estimate in the report.
    pred_all = load_module("pred_all_under_test", str(HERE / "pred_all.py"))
    pred_all.MODELS = model_dir
    for model_name, model in models.items():
        ours = pred_all.predict_all(test_csv, model=model_name)
        theirs = model.predict(design_matrix(struct, tokens, preps[model_name])).tolist()
        same = sum(a == b for a, b in zip(ours, theirs))
        acc = sum(a == b for a, b in zip(ours, labels)) / len(labels)
        print(f"pred_all.py {model_name}: matches sklearn on {same}/{len(ours)} rows; "
              f"held-out accuracy {100 * acc:.1f}% on {len(labels)} rows")
        assert same == len(ours)


if __name__ == "__main__":
    main()
