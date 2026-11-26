# Submission files

Everything that goes to MarkUs, copied from `main`. Nothing else in the repository is submitted.

| MarkUs assignment | Upload | What it is |
| --- | --- | --- |
| Project prediction | `prediction/pred.py` and `prediction/weights.json` | `predict_all(filename)` for the final model, Bernoulli naive Bayes (standard library, numpy and pandas only), and its weights, trained on all 861 cleaned rows |
| Project report | `code.zip` | The 11 scripts behind the report, also unzipped in `code/` |
| Project report | `report.pdf` | Not added yet |

## What is in `code.zip`, in reading order

1. `methodology/features.py`: cleaning, the 70/30 split by respondent, features, and preprocessing fitted on training rows only.
2. `data_exploration/missing_values.py`: Table 1, missing values.
3. `data_exploration/preprocessing_checks.py`: Table 1, price parsing, outliers and the sparse-text numbers. It imports the feature code in `methodology/baseline.py`.
4. `data_exploration/eda_with_figures.py`: Figure 1.
5. `data_exploration/eda.py`: Figure 2, mutual information and `split_ids.csv`.
6. `methodology/model_selection.py`: Tables 2–4, which cover grouped 5-fold cross-validation for all three families, the tie rule and the single test-set evaluation.
7. `methodology/train_final.py`: refits the final model and writes `weights.json`.
8. `methodology/submission/pred.py`: the submitted prediction script (the same file as `prediction/pred.py`).
9. `methodology/pred_all.py` and `methodology/check_pred.py`: numpy versions of all three models, and the check that `pred.py` matches scikit-learn and meets the 10 MB and time limits.

The files in `code/` are copies for submission. The working versions, which run from the repository, are in `data_exploration/` and `methodology/`. If those change, copy them here again and rebuild `code.zip` from inside `code/`.
