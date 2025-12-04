import pickle
import joblib
from pathlib import Path
import sys

models_dir = Path("models")

for model_file in models_dir.glob("*.pkl"):
    print(f"Testing {model_file.name}...")
    try:
        with open(model_file, "rb") as f:
            pickle.load(f)
        print(f"  SUCCESS (pickle)")
    except Exception as e_pickle:
        print(f"  FAILED (pickle): {e_pickle}")
        try:
            joblib.load(model_file)
            print(f"  SUCCESS (joblib)")
        except Exception as e_joblib:
            print(f"  FAILED (joblib): {e_joblib}")
