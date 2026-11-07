Proposal: https://docs.google.com/document/d/125V\_es4TMSLcpcNPxWSrk7r09S27686kTpooYsT3vPo/edit?tab=t.0


## Submission files

Prediction (MarkUs project prediction assignment), upload these four files together:

| File | What it is |
| --- | --- |
| `pred.py` | `predict_all(filename)`; standard library, numpy and pandas only |
| `features.py` | Cleaning and feature extraction used by `pred.py` |
| `model_params.npz` | Trained parameters of the three models compared in the report |
| `model_meta.json` | Class names, preprocessing statistics and vocabularies |

`pred.py` uses Bernoulli naive Bayes, the final model (`MODEL = "naive_bayes"`); the random
forest (`"random_forest"`) and kNN (`"knn"`) are the other two families in the report. All
three are fit on the instructors' training + validation respondents (609 rows).

Report (MarkUs project report assignment): `report.pdf` from the LaTeX template, and
`code.zip`, built with `python make_code_zip.py`.

Model selection for the report: `python model_selection.py` (results in
`model_selection_results/`). To rebuild the model files: `python train_final.py` (needs
scikit-learn). To check `pred.py` against scikit-learn, the test rows, the 60-row time
limit and the 10 MB limit: `python check_pred.py`.

## Where each number in the report comes from

Every log in `logs/` is the full output of one run of the script named.

| Report item | Code | Output / log |
| --- | --- | --- |
| Data split (146/57/86 respondents) | `features.py` (`respondent_split`) | `split_ids.csv` |
| Table 1: missing values, price parsing | `missing_values.py`, `preprocessing_checks.py` | `logs/missing_values.log`, `logs/preprocessing_checks.log`, `eda_outputs/` |
| Figures 1 and 2, mutual information | `eda_with_figures.py`, `eda.py` | `logs/eda_with_figures.log`, `logs/eda.log`, `eda_outputs/mutual_information.csv` |
| Preprocessing (imputation, scaling, vocabulary, top words) | `features.py` (`fit_preprocessing`, `design_matrix`) | |
| Table 2: grouped 5-fold CV over all 400 hyperparameter settings | `model_selection.py` | `logs/model_selection.log`, `model_selection_results/cv_all.csv` |
| Table 3: tuned models on validation | `model_selection.py` | `logs/model_selection.log`, `model_selection_results/summary.csv`, `model_selection_results/val_confusion.json` |
| Table 4 and the test result | `model_selection.py` (`final_test`) | `logs/model_selection.log`, `model_selection_results/final_test.json` |
| Final models and `pred.py` checks | `train_final.py`, `pred.py`, `check_pred.py` | `logs/check_pred.log` |
| Earlier sweep of every model in the course notes, split ratios, synthetic data | `model_sweep/` | CSVs in `model_sweep/` |
