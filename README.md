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
