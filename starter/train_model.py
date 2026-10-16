"""Script to train the census salary model.

Run from the project root with::

    python -m starter.train_model

The script loads and processes the data, trains the model, evaluates it on a
held-out test split, saves the model, the one hot encoder and the label
binarizer, and writes the per-slice metrics to ``slice_output.txt``.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.config import (
    CAT_FEATURES,
    DATA_PATH,
    ENCODER_PATH,
    LABEL,
    LB_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SLICE_OUTPUT_PATH,
    TEST_SIZE,
)
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    format_slice_metrics,
    inference,
    save_artifact,
    train_model,
)

PathLike = Union[str, Path]


def load_data(path: PathLike = DATA_PATH) -> pd.DataFrame:
    """Read the (space-free) census csv. Column names are left untouched."""
    return pd.read_csv(path, skipinitialspace=True)


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test sets (80/20, reproducible).

    Stratifying keeps the share of >50K rows equal in both splits.
    """
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[LABEL],
    )
    return train, test


def run_training(
    data_path: PathLike = DATA_PATH,
    model_path: PathLike = MODEL_PATH,
    encoder_path: PathLike = ENCODER_PATH,
    lb_path: PathLike = LB_PATH,
    slice_output_path: PathLike = SLICE_OUTPUT_PATH,
) -> Dict[str, float]:
    """Train, evaluate and save the model; write the slice report.

    Returns the overall precision, recall and fbeta on the test split.
    """
    data = load_data(data_path)

    train, test = split_data(data)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    precision, recall, fbeta = compute_model_metrics(
        y_test, inference(model, X_test)
    )

    save_artifact(model, model_path)
    save_artifact(encoder, encoder_path)
    save_artifact(lb, lb_path)

    report_lines = [
        "Model performance on slices of the held-out test set",
        f"Test rows: {len(test)} | training rows: {len(train)}",
        f"Overall | precision={precision:.4f} | recall={recall:.4f}"
        f" | fbeta={fbeta:.4f}",
        "Slices: the value of one categorical feature is held fixed.",
        "positives = rows whose true label is >50K.",
        "Caution: metrics use zero_division=1. If a slice has no predicted",
        "(or no actual) >50K rows, precision (or recall) is reported as 1.0",
        "by convention rather than measured. Check n and positives first.",
        "",
    ]
    for feature in CAT_FEATURES:
        report_lines.append(f"Feature: {feature}")
        results = compute_slice_metrics(
            test,
            feature,
            CAT_FEATURES,
            LABEL,
            model,
            encoder,
            lb,
            # Every distinct value in the full data set gets a line, even
            # a rare value that happens to be absent from the test split.
            values=sorted(data[feature].unique()),
        )
        report_lines.append(format_slice_metrics(results))
        report_lines.append("")
    report = "\n".join(report_lines)

    Path(slice_output_path).write_text(report)
    print(report)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fbeta": float(fbeta),
    }


def main() -> None:
    """Entry point for ``python -m starter.train_model``."""
    run_training()


if __name__ == "__main__":
    main()
