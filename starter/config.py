"""Shared constants: feature names, label and artifact locations.

All paths are derived from this file's location so that they resolve
identically on a developer machine, in CI and on the hosting platform.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "encoder.joblib"
LB_PATH = MODEL_DIR / "lb.joblib"
SLICE_OUTPUT_PATH = PROJECT_ROOT / "slice_output.txt"

LABEL = "salary"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Model input columns in the same order as the columns of data/census.csv
# (everything except the label). The API uses this to build the data frame
# that is handed to ``process_data`` so that inference sees exactly the
# column layout the model was trained on.
FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
