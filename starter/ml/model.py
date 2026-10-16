"""Model training, inference, evaluation, persistence and slice metrics."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # min_samples_leaf keeps the trees small (about 10 MB on disk instead of
    # about 70 MB for fully grown trees) and acts as a regulariser. The fixed
    # random_state makes training reproducible.
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifact(artifact: Any, path: Union[str, Path]) -> Path:
    """Serialise a model, encoder or label binarizer to ``path``.

    Parent directories are created when they do not exist. Returns the path
    that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Union[str, Path]) -> Any:
    """Load a model, encoder or label binarizer saved by ``save_artifact``."""
    return joblib.load(Path(path))


def compute_slice_metrics(
    df: pd.DataFrame,
    feature: str,
    categorical_features: Sequence[str],
    label: str,
    model,
    encoder,
    lb,
    values: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Compute model metrics with the value of ``feature`` held fixed.

    One result is returned for every value in ``values`` (by default every
    distinct value of ``feature`` found in ``df``). A value that has no rows
    in ``df`` still gets a result, with ``n`` equal to 0 and no metrics, so
    that the output covers every distinct value.

    Inputs
    ------
    df : pd.DataFrame
        Raw (unprocessed) data that contains the feature and the label.
    feature : str
        Name of the column whose value is held fixed.
    categorical_features : Sequence[str]
        Categorical columns, passed through to ``process_data``.
    label : str
        Name of the label column.
    model, encoder, lb :
        Fitted model, one hot encoder and label binarizer.
    values : Sequence, optional
        Slice values to report on.
    Returns
    -------
    results : list[dict]
        Keys: feature, value, n, positives, precision, recall, fbeta.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' is not a column of the data.")

    if values is None:
        values = sorted(df[feature].unique())

    results: List[Dict[str, Any]] = []
    for value in values:
        subset = df[df[feature] == value]
        result: Dict[str, Any] = {
            "feature": feature,
            "value": value,
            "n": int(len(subset)),
            "positives": 0,
            "precision": None,
            "recall": None,
            "fbeta": None,
        }
        if len(subset) > 0:
            X_slice, y_slice, _, _ = process_data(
                subset,
                categorical_features=list(categorical_features),
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            precision, recall, fbeta = compute_model_metrics(
                y_slice, inference(model, X_slice)
            )
            result["positives"] = int(np.sum(y_slice))
            result["precision"] = float(precision)
            result["recall"] = float(recall)
            result["fbeta"] = float(fbeta)
        results.append(result)
    return results


def format_slice_metrics(results: Sequence[Dict[str, Any]]) -> str:
    """Render the output of ``compute_slice_metrics`` as plain text."""
    lines: List[str] = []
    for result in results:
        prefix = f"{result['feature']} = {result['value']}"
        if result["n"] == 0:
            lines.append(
                f"{prefix} | n=0 | no test samples, metrics undefined"
            )
            continue
        lines.append(
            f"{prefix} | n={result['n']} | positives={result['positives']}"
            f" | precision={result['precision']:.4f}"
            f" | recall={result['recall']:.4f}"
            f" | fbeta={result['fbeta']:.4f}"
        )
    return "\n".join(lines)
