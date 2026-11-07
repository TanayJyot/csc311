Proposal: https://docs.google.com/document/d/125V\_es4TMSLcpcNPxWSrk7r09S27686kTpooYsT3vPo/edit?tab=t.0


## Submission files

Prediction (MarkUs project prediction assignment), upload these four files together:

| File | What it is |
| --- | --- |
| `pred.py` | `predict_all(filename)`; standard library, numpy and pandas only |
| `features.py` | Cleaning and feature extraction used by `pred.py` |
| `model_params.npz` | Trained parameters of the three final models |
| `model_meta.json` | Class names, preprocessing statistics and vocabularies |

`pred.py` uses the random forest by default (`MODEL = "random_forest"`); Bernoulli naive
Bayes (`"bernoulli_nb"`) and L1 softmax regression (`"softmax"`) are the two runner-ups.

Report (MarkUs project report assignment): `report.pdf` from the LaTeX template, and
`code.zip`, built with `python make_code_zip.py`.

To rebuild the model files: `python train_final.py` (needs scikit-learn). To check
`pred.py` against scikit-learn, the held-out test rows, the 60-row time limit and the
10 MB limit: `python check_pred.py`, then `python train_final.py` again.
