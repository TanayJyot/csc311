"""check_pred.py - check pred.py against the MarkUs rules and against scikit-learn.

1. Fits the three models on the instructors' train+val respondents only, writes a
   held-out test CSV without the label column, and runs pred.predict_all on it
   from a clean folder holding only the files we would submit.
2. Checks the numpy predictions equal the scikit-learn ones, reports held-out
   accuracy, the time for 60 rows and the size of the submitted files.

Run: python check_pred.py   (needs scikit-learn; run train_final.py afterwards
     to re-export the models fit on all rows)
"""
import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import time

import pandas as pd

from features import (TEXT_COLS, blank_rows, design_matrix, load_survey,
                      respondent_split, structured_features, tokenize)
from train_final import DATA, fit_and_export

SUBMITTED = ["pred.py", "features.py", "model_params.npz", "model_meta.json"]
ALLOWED_IMPORTS = {"json", "os", "sys", "re", "numpy", "pandas", "features"}


def check_imports():
    """pred.py and features.py may import only the standard library, numpy and pandas."""
    for name in ("pred.py", "features.py"):
        for node in ast.walk(ast.parse(open(name).read())):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] in ALLOWED_IMPORTS, f"{name} imports {module}"


def main():
    check_imports()
    folder = tempfile.mkdtemp()
    models, preps = fit_and_export(folder, holdout=True)
    for name in ("pred.py", "features.py"):
        shutil.copy(name, folder)

    raw = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    short = load_survey(DATA)
    keep = ~blank_rows(short)
    split = pd.Series("dropped", index=short.index)
    split[keep] = respondent_split(short.loc[keep, "id"])
    test_mask = split == "test"
    test_csv = os.path.join(folder, "test.csv")
    raw.loc[test_mask].drop(columns=["label"]).to_csv(test_csv, index=False)
    labels = short.loc[test_mask, "label"].tolist()

    spec = importlib.util.spec_from_file_location("pred_under_test", os.path.join(folder, "pred.py"))
    sys.path.insert(0, folder)
    pred = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pred)

    test = short.loc[test_mask].reset_index(drop=True)
    struct = structured_features(test).drop(columns=["price"])
    tokens = {c: test[c].map(tokenize) for c in TEXT_COLS}
    for model_name, model in models.items():
        ours = pred.predict_all(test_csv, model=model_name)
        theirs = model.predict(design_matrix(struct, tokens, preps[model_name])).tolist()
        same = sum(a == b for a, b in zip(ours, theirs))
        acc = sum(a == b for a, b in zip(ours, labels)) / len(labels)
        print(f"{model_name}: matches sklearn on {same}/{len(ours)} rows; "
              f"held-out accuracy {100 * acc:.1f}% on {len(labels)} rows")
        assert same == len(ours)

    small = os.path.join(folder, "sixty.csv")
    raw.loc[test_mask].head(60).drop(columns=["label"]).to_csv(small, index=False)
    start = time.time()
    pred.predict_all(small)
    print(f"60 predictions in {time.time() - start:.2f} s")
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in SUBMITTED)
    print(f"submitted files: {size / 1e6:.2f} MB (limit 10 MB)")


if __name__ == "__main__":
    main()
