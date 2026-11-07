"""make_code_zip.py - build code.zip for the report submission.

code.zip holds every .py file used to develop the final model (repository root and
model_sweep/), without the data, generated outputs or the saved model files.
Run: python make_code_zip.py
"""
import glob
import zipfile

files = sorted(glob.glob("*.py") + glob.glob("model_sweep/*.py") + glob.glob("model_sweep/synthetic/*.py"))
with zipfile.ZipFile("code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(name)
print(f"code.zip: {len(files)} files")
