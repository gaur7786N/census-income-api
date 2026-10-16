"""Tests for the training script (starter/train_model.py)."""

from starter.config import CAT_FEATURES, DATA_PATH, FEATURE_COLUMNS
from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact
from starter.train_model import load_data, run_training, split_data


def test_load_data_keeps_original_column_names():
    data = load_data(DATA_PATH)
    assert "education-num" in data.columns
    assert "native-country" in data.columns
    assert data.shape == (32561, 15)
    # The cleaned csv has no stray spaces in text columns.
    assert not data["workclass"].str.contains(" ").any()


def test_split_data_is_disjoint_stratified_and_80_20():
    data = load_data(DATA_PATH)
    train, test = split_data(data)
    assert len(train) + len(test) == len(data)
    assert abs(len(test) / len(data) - 0.20) < 0.001
    # No row may appear in both splits (that would leak test data).
    assert set(train.index).isdisjoint(test.index)
    # Stratified: the share of >50K rows is the same in both splits.
    share = (data["salary"] == ">50K").mean()
    assert abs((train["salary"] == ">50K").mean() - share) < 0.001
    assert abs((test["salary"] == ">50K").mean() - share) < 0.001


def test_split_data_is_reproducible():
    data = load_data(DATA_PATH)
    first_train, first_test = split_data(data)
    second_train, second_test = split_data(data)
    assert first_train.index.equals(second_train.index)
    assert first_test.index.equals(second_test.index)


def test_run_training_saves_artifacts_and_slice_report(tmp_path):
    metrics = run_training(
        model_path=tmp_path / "model.joblib",
        encoder_path=tmp_path / "encoder.joblib",
        lb_path=tmp_path / "lb.joblib",
        slice_output_path=tmp_path / "slice_output.txt",
    )

    for name in ("precision", "recall", "fbeta"):
        assert 0.5 < metrics[name] <= 1.0

    for name in ("model.joblib", "encoder.joblib", "lb.joblib"):
        assert (tmp_path / name).stat().st_size > 0

    # The saved artifacts must work together for inference.
    model = load_artifact(tmp_path / "model.joblib")
    encoder = load_artifact(tmp_path / "encoder.joblib")
    lb = load_artifact(tmp_path / "lb.joblib")
    data = load_data(DATA_PATH)
    X, _, _, _ = process_data(
        data.head(1)[FEATURE_COLUMNS],
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert inference(model, X).shape == (1,)

    # One block per categorical feature, one line per distinct value.
    report = (tmp_path / "slice_output.txt").read_text()
    for feature in CAT_FEATURES:
        assert f"Feature: {feature}" in report
        for value in data[feature].unique():
            assert f"{feature} = {value} | n=" in report
