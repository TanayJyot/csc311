"""make_code_zip.py - refresh the submissions/ folder from the working files.

submissions/ holds exactly what is uploaded to MarkUs:
  prediction/pred.py, prediction/weights.json   copies of methodology/prediction/
  code/                                        the scripts behind the report
  code.zip                                     code/ zipped (no data, no model files)

Run: python methodology/make_code_zip.py   (from any folder)
"""
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submissions")

# The scripts behind the report, in reading order.
CODE = [
    "methodology/features.py",
    "data_exploration/missing_values.py",
    "data_exploration/preprocessing_checks.py",
    "methodology/baseline.py",              # imported by preprocessing_checks.py
    "data_exploration/eda_with_figures.py",
    "data_exploration/eda.py",
    "methodology/model_selection.py",
    "methodology/train_final.py",
    "methodology/prediction/pred.py",
    "methodology/pred_all.py",
    "methodology/check_pred.py",
]
PREDICTION = ["methodology/prediction/pred.py", "methodology/prediction/weights.json"]

shutil.rmtree(os.path.join(OUT, "code"), ignore_errors=True)
for name in CODE:
    dest = os.path.join(OUT, "code", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, name), dest)

os.makedirs(os.path.join(OUT, "prediction"), exist_ok=True)
for name in PREDICTION:
    shutil.copy2(os.path.join(ROOT, name), os.path.join(OUT, "prediction", os.path.basename(name)))

with zipfile.ZipFile(os.path.join(OUT, "code.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(CODE):
        z.write(os.path.join(OUT, "code", name), name)
print(f"submissions/: code.zip with {len(CODE)} files, prediction/ with {len(PREDICTION)} files")
