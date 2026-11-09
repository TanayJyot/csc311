Proposal: https://docs.google.com/document/d/125V\_es4TMSLcpcNPxWSrk7r09S27686kTpooYsT3vPo/edit?tab=t.0

## Layout

| Folder | What is in it |
| --- | --- |
| `data_exploration/` | Exploration scripts, the train/val/test split (`split_ids.csv`), output tables in `eda_outputs/`, run logs in `logs/` |
| `methodology/` | Model selection, final training, checks, the earlier model sweep (`model_sweep/`), run logs in `logs/` |
| `methodology/submission/` | Exactly the files uploaded to MarkUs for prediction |

The data, `csc311_project_training_202609.csv`, stays at the repository root. Every
script finds it and `features.py` on its own, so it can be run from any folder, for
example `python methodology/model_selection.py`.

## Submission files

Prediction (MarkUs project prediction assignment): upload the four files in
`methodology/submission/` together.

| File | What it is |
| --- | --- |
| `pred.py` | `predict_all(filename)`; standard library, numpy and pandas only |
| `features.py` | Cleaning, feature extraction and preprocessing used by `pred.py` |
| `model_params.npz` | Trained parameters of the three models compared in the report |
| `model_meta.json` | Class names, preprocessing statistics and vocabularies |

`pred.py` uses Bernoulli naive Bayes, the final model (`MODEL = "naive_bayes"`); the random
forest (`"random_forest"`) and kNN (`"knn"`) are the other two families in the report. All
three are fit on the 70% development set (201 respondents, 603 rows).

Report (MarkUs project report assignment): `report.pdf` from the LaTeX template, and
`code.zip`, built with `python methodology/make_code_zip.py`.

Model selection for the report: `python methodology/model_selection.py` (results in
`methodology/model_selection_results/`). To rebuild the model files:
`python methodology/train_final.py` (needs scikit-learn). To check `pred.py` against
scikit-learn, the test rows, the 60-row time limit and the 10 MB limit:
`python methodology/check_pred.py`.

## Where each number in the report comes from

Every log in a `logs/` folder is the full output of one run of the script named. All
exploration uses the 70% development set only. The
exploration logs were run with `PYTHONHASHSEED=0`, which fixes the order of tied words.

| Report item | Code | Output / log |
| --- | --- | --- |
| Cleaning (7 blank rows, then respondents without one row per sport: 861 rows, 287 respondents) and the 70/30 split (development: 201 respondents, 603 rows; test: 86 respondents, 258 rows) | `methodology/submission/features.py` (`clean_survey`, `dev_test_split`) | `data_exploration/split_ids.csv`, `data_exploration/logs/eda.log` |
| Table 1: missing values, price parsing | `data_exploration/missing_values.py`, `data_exploration/preprocessing_checks.py` | `data_exploration/logs/missing_values.log`, `data_exploration/logs/preprocessing_checks.log`, `data_exploration/eda_outputs/` |
| Figures 1 and 2, mutual information | `data_exploration/eda_with_figures.py`, `data_exploration/eda.py` | `data_exploration/logs/eda_with_figures.log`, `data_exploration/logs/eda.log`, `data_exploration/eda_outputs/mutual_information.csv` |
| Preprocessing (imputation, scaling, vocabulary, top words) | `methodology/submission/features.py` (`fit_preprocessing`, `design_matrix`) | |
| Table 2: grouped 5-fold CV on the 70% development set over all 400 hyperparameter settings | `methodology/model_selection.py` | `methodology/logs/model_selection.log`, `methodology/model_selection_results/cv_all.csv` |
| Table 3: tuned models compared on the CV results (accuracy, per-sport F1, confusion matrix) | `methodology/model_selection.py` | `methodology/logs/model_selection.log`, `methodology/model_selection_results/summary.csv`, `methodology/model_selection_results/cv_confusion.json` |
| Table 4 and the test result (252 rows) | `methodology/model_selection.py` (`main`, step 4) | `methodology/logs/model_selection.log`, `methodology/model_selection_results/final_test.json` |
| Final models and `pred.py` checks | `methodology/train_final.py`, `methodology/submission/pred.py`, `methodology/check_pred.py` | `methodology/logs/check_pred.log` |
| Pipeline check quoted in the proposal | `methodology/baseline.py` | |
| Earlier sweep of every model in the course notes, split ratios, synthetic data | `methodology/model_sweep/` | CSVs in `methodology/model_sweep/` |
