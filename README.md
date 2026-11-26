Proposal: https://docs.google.com/document/d/125V\_es4TMSLcpcNPxWSrk7r09S27686kTpooYsT3vPo/edit?tab=t.0

## How to read the code (in order)

1. `methodology/features.py`: everything shared. Loading, `clean_survey`
   (drop blank rows and incomplete respondents), `dev_test_split` (70/30 by respondent),
   answer parsing, `tokenize`, and `fit_preprocessing` / `design_matrix`, which learn
   medians, scaling and the vocabulary from training rows only.
2. `data_exploration/missing_values.py` and `data_exploration/preprocessing_checks.py`:
   Table 1 (missing values, price parsing and outliers), on the development set only.
3. `data_exploration/eda_with_figures.py` (Figure 1) and `data_exploration/eda.py`
   (Figure 2, mutual information, `split_ids.csv`): Data Exploration.
4. `methodology/model_selection.py`: Methodology and Results. Grids for kNN, random
   forest and naive Bayes (Table 2), grouped 5-fold CV on the development set, the
   one-standard-error tie rule (Table 3), then one test-set evaluation (Table 4).
5. `methodology/evaluation_checks.py`: checks that the ~92% is not leakage (shuffled
   labels, one question at a time, 200 other splits, nested CV).
6. `methodology/train_final.py`: refits the tuned settings on the development set and
   saves `models/` (all three models) and `submission/weights.json` (naive Bayes only).
7. `methodology/submission/pred.py`: the submitted script. Self-contained (its own copy of
   the cleaning and feature code), naive Bayes only, numpy and pandas only.
8. `methodology/pred_all.py`: numpy versions of all three families, used during testing.
9. `methodology/check_pred.py`: confirms `pred.py` matches scikit-learn and `features.py`,
   the time limit and the 10 MB limit, and that `pred_all.py` matches for all three models.

Background only (not used for the final choice): `methodology/baseline.py` (proposal
sanity check) and `methodology/model_sweep/` (early exploration of more model families).

## Layout

| Folder | What is in it |
| --- | --- |
| `data_exploration/` | Exploration scripts, the 70/30 development/test split (`split_ids.csv`), output tables in `eda_outputs/`, run logs in `logs/` |
| `methodology/` | Shared feature code (`features.py`), model selection, final training, `pred_all.py` and the three saved models (`models/`), checks, the earlier model sweep (`model_sweep/`), run logs in `logs/` |
| `methodology/submission/` | Exactly the files uploaded to MarkUs for prediction |

The data, `csc311_project_training_202609.csv`, stays at the repository root. Every
script finds it and `features.py` on its own, so it can be run from any folder, for
example `python methodology/model_selection.py`.

## Submission files

Prediction (MarkUs project prediction assignment): upload the two files in
`methodology/submission/` together.

| File | What it is |
| --- | --- |
| `pred.py` | `predict_all(filename)` for the final model, Bernoulli naive Bayes. Contains all cleaning and feature code; standard library, numpy and pandas only |
| `weights.json` | What was learned from the 70% development set (201 respondents, 603 rows): preprocessing statistics, the 100 kept words, and the naive Bayes log-probabilities |

`methodology/pred_all.py` (with `methodology/models/`) is the development version that also
runs the random forest and kNN; it is not submitted.

Report (MarkUs project report assignment): `report.pdf` from the LaTeX template, and
`code.zip`, built with `python methodology/make_code_zip.py`.

Model selection for the report: `python methodology/model_selection.py` (results in
`methodology/model_selection_results/`). To rebuild `weights.json` and `models/`:
`python methodology/train_final.py` (needs scikit-learn). To check `pred.py` against
scikit-learn, the test rows, the 60-row time limit and the 10 MB limit:
`python methodology/check_pred.py`.

## Where each number in the report comes from

Every log in a `logs/` folder is the full output of one run of the script named. All
exploration uses the 70% development set only. The
exploration logs were run with `PYTHONHASHSEED=0`, which fixes the order of tied words.

| Report item | Code | Output / log |
| --- | --- | --- |
| Cleaning (7 blank rows, then respondents without one row per sport: 861 rows, 287 respondents) and the 70/30 split (development: 201 respondents, 603 rows; test: 86 respondents, 258 rows) | `methodology/features.py` (`clean_survey`, `dev_test_split`) | `data_exploration/split_ids.csv`, `data_exploration/logs/eda.log` |
| Table 1: missing values, price parsing | `data_exploration/missing_values.py`, `data_exploration/preprocessing_checks.py` | `data_exploration/logs/missing_values.log`, `data_exploration/logs/preprocessing_checks.log`, `data_exploration/eda_outputs/` |
| Figures 1 and 2, mutual information | `data_exploration/eda_with_figures.py`, `data_exploration/eda.py` | `data_exploration/logs/eda_with_figures.log`, `data_exploration/logs/eda.log`, `data_exploration/eda_outputs/mutual_information.csv` |
| Preprocessing (imputation, scaling, vocabulary, top words) | `methodology/features.py` (`fit_preprocessing`, `design_matrix`) | |
| Table 2: grouped 5-fold CV on the 70% development set over all 400 hyperparameter settings | `methodology/model_selection.py` | `methodology/logs/model_selection.log`, `methodology/model_selection_results/cv_all.csv` |
| Table 3: tuned models compared on the CV results (accuracy, per-sport F1, confusion matrix) | `methodology/model_selection.py` | `methodology/logs/model_selection.log`, `methodology/model_selection_results/summary.csv`, `methodology/model_selection_results/cv_confusion.json` |
| Table 4 and the test result (258 rows) | `methodology/model_selection.py` (`main`, step 4) | `methodology/logs/model_selection.log`, `methodology/model_selection_results/final_test.json` |
| Final models and `pred.py` checks | `methodology/train_final.py`, `methodology/submission/pred.py`, `methodology/pred_all.py`, `methodology/check_pred.py` | `methodology/logs/check_pred.log` |
| Leakage checks on the evaluation (shuffled labels, one question at a time, 200 other splits, nested CV) | `methodology/evaluation_checks.py` | |
| Pipeline check quoted in the proposal | `methodology/baseline.py` | |
| Earlier sweep of every model in the course notes, split ratios, synthetic data | `methodology/model_sweep/` | CSVs in `methodology/model_sweep/` |
