"""Unit tests for the functions in starter/ml/data.py and starter/ml/model.py.

Most tests use a tiny synthetic data set so they are fast and do not depend
on a trained artifact. Training is stochastic, so the tests check types,
shapes, value ranges and exact values for deterministic functions rather
than exact predictions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.config import (
    CAT_FEATURES,
    ENCODER_PATH,
    FEATURE_COLUMNS,
    LB_PATH,
    MODEL_PATH,
)
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    format_slice_metrics,
    inference,
    load_artifact,
    save_artifact,
    train_model,
)

TINY_CAT = ["workclass", "sex"]


@pytest.fixture
def tiny_data() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 120
    age = rng.randint(18, 70, size=n)
    workclass = rng.choice(["Private", "Gov", "Self"], size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    # A label that depends on the features, so a model can learn something.
    salary = np.where((age > 40) & (workclass != "Self"), ">50K", "<=50K")
    return pd.DataFrame(
        {"age": age, "workclass": workclass, "sex": sex, "salary": salary}
    )


@pytest.fixture
def fitted(tiny_data):
    X, y, encoder, lb = process_data(
        tiny_data, categorical_features=TINY_CAT, label="salary",
        training=True,
    )
    return tiny_data, X, y, encoder, lb, train_model(X, y)


def test_process_data_training_returns_expected_types(tiny_data):
    X, y, encoder, lb = process_data(
        tiny_data, categorical_features=TINY_CAT, label="salary",
        training=True,
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    # 1 continuous column + 3 workclass values + 2 sex values.
    assert X.shape == (len(tiny_data), 1 + 3 + 2)
    assert y.shape == (len(tiny_data),)
    assert set(np.unique(y)) == {0, 1}


def test_process_data_inference_reuses_fitted_encoder(tiny_data):
    _, _, encoder, lb = process_data(
        tiny_data, categorical_features=TINY_CAT, label="salary",
        training=True,
    )
    unlabeled = tiny_data.drop(columns=["salary"]).head(5)
    X, y, enc_out, lb_out = process_data(
        unlabeled, categorical_features=TINY_CAT, label=None,
        training=False, encoder=encoder, lb=lb,
    )
    assert X.shape == (5, 6)
    assert y.size == 0
    assert enc_out is encoder
    assert lb_out is lb


def test_train_model_returns_fitted_random_forest(fitted):
    _, X, y, _, _, model = fitted
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")
    assert model.n_features_in_ == X.shape[1]
    assert list(model.classes_) == [0, 1]


def test_inference_returns_binary_predictions_for_each_row(fitted):
    _, X, _, _, _, model = fitted
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_learns_the_synthetic_rule(fitted):
    _, X, y, _, _, model = fitted
    precision, recall, fbeta = compute_model_metrics(y, inference(model, X))
    # Training-set check on an easy rule: a sanity check, not a benchmark.
    assert fbeta > 0.9


def test_compute_model_metrics_returns_known_values():
    # TP=2, FP=0, FN=1, so precision, recall and F1 all differ and a swap of
    # any two of them would be noticed.
    y = np.array([1, 1, 1, 0, 0, 0])
    preds = np.array([1, 1, 0, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
    assert isinstance(float(precision), float)


def test_compute_model_metrics_perfect_predictions():
    y = np.array([1, 0, 1, 0])
    assert compute_model_metrics(y, y) == (1.0, 1.0, 1.0)


def test_compute_slice_metrics_reports_every_requested_value(fitted):
    data, _, _, encoder, lb, model = fitted
    values = ["Private", "Gov", "Self", "Never-worked"]
    results = compute_slice_metrics(
        data, "workclass", TINY_CAT, "salary", model, encoder, lb,
        values=values,
    )
    assert [r["value"] for r in results] == values
    for result in results[:3]:
        expected_n = int((data["workclass"] == result["value"]).sum())
        assert result["n"] == expected_n
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["fbeta"] <= 1.0
    # A value that is absent from the data is reported, without metrics.
    assert results[3]["n"] == 0
    assert results[3]["precision"] is None


def test_compute_slice_metrics_defaults_to_values_present(fitted):
    data, _, _, encoder, lb, model = fitted
    results = compute_slice_metrics(
        data, "sex", TINY_CAT, "salary", model, encoder, lb
    )
    assert sorted(r["value"] for r in results) == ["Female", "Male"]
    assert sum(r["n"] for r in results) == len(data)


def test_compute_slice_metrics_rejects_unknown_feature(fitted):
    data, _, _, encoder, lb, model = fitted
    with pytest.raises(KeyError):
        compute_slice_metrics(
            data, "height", TINY_CAT, "salary", model, encoder, lb
        )


def test_format_slice_metrics_lists_each_slice():
    results = [
        {"feature": "sex", "value": "Male", "n": 10, "positives": 4,
         "precision": 0.5, "recall": 0.25, "fbeta": 1 / 3},
        {"feature": "sex", "value": "Other", "n": 0, "positives": 0,
         "precision": None, "recall": None, "fbeta": None},
    ]
    lines = format_slice_metrics(results).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("sex = Male | n=10 | positives=4")
    assert "precision=0.5000" in lines[0]
    assert "recall=0.2500" in lines[0]
    assert "fbeta=0.3333" in lines[0]
    assert lines[1] == "sex = Other | n=0 | no test samples, metrics undefined"


def test_save_and_load_artifact_round_trip(fitted, tmp_path):
    _, X, _, _, _, model = fitted
    path = save_artifact(model, tmp_path / "nested" / "model.joblib")
    assert path.exists()
    loaded = load_artifact(path)
    assert np.array_equal(inference(loaded, X), inference(model, X))


def test_committed_artifacts_run_inference_without_retraining():
    # Guards the submission requirement: a fresh clone must be able to load
    # the model, encoder and label binarizer and predict immediately.
    model = load_artifact(MODEL_PATH)
    encoder = load_artifact(ENCODER_PATH)
    lb = load_artifact(LB_PATH)
    row = pd.DataFrame(
        [[39, "State-gov", 77516, "Bachelors", 13, "Never-married",
          "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40,
          "United-States"]],
        columns=FEATURE_COLUMNS,
    )
    X, _, _, _ = process_data(
        row, categorical_features=CAT_FEATURES, label=None, training=False,
        encoder=encoder, lb=lb,
    )
    prediction = lb.inverse_transform(inference(model, X))
    assert list(prediction) in (["<=50K"], [">50K"])
