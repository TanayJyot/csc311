"""make_code_zip.py - build code.zip for the report submission.

code.zip holds every .py file used to develop the final model (data_exploration/ and
methodology/), without the data, generated outputs or the saved model files.
Run: python methodology/make_code_zip.py   (writes code.zip at the repository root)
"""
import glob
import os
import zipfile

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root

files = sorted(glob.glob("**/*.py", recursive=True))
with zipfile.ZipFile("code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(name)
print(f"code.zip: {len(files)} files")
