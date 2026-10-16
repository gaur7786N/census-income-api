"""Tests for the FastAPI application (GET and POST endpoints)."""

from fastapi.testclient import TestClient

import numpy as np
import pandas as pd

from main import CensusRecord, app, encoder, lb, record_to_features
from starter.config import CAT_FEATURES, DATA_PATH, FEATURE_COLUMNS
from starter.ml.data import process_data

client = TestClient(app)

# A person the saved model classifies as earning >50K (P(>50K) is about 0.97).
HIGH_INCOME_PERSON = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 160187,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}

# A person the saved model classifies as earning <=50K (P(>50K) is ~0.00).
LOW_INCOME_PERSON = {
    "age": 23,
    "workclass": "Private",
    "fnlgt": 122272,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
}


def test_get_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": (
            "Welcome to the Census Income Prediction API. "
            "Send a POST request to /predict to classify a person."
        )
    }


def test_post_predicts_salary_above_50k():
    response = client.post("/predict", json=HIGH_INCOME_PERSON)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_post_predicts_salary_at_most_50k():
    response = client.post("/predict", json=LOW_INCOME_PERSON)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_rejects_body_with_missing_field():
    incomplete = dict(LOW_INCOME_PERSON)
    del incomplete["education-num"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert "education-num" in str(response.json()["detail"])


def test_post_rejects_wrong_type():
    invalid = dict(LOW_INCOME_PERSON, age="not-a-number")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert "age" in str(response.json()["detail"])


def test_post_handles_unseen_category_without_error():
    # The encoder ignores categories it did not see in training, so a new
    # native country must not crash the API.
    unseen = dict(LOW_INCOME_PERSON, **{"native-country": "Atlantis"})
    response = client.post("/predict", json=unseen)
    assert response.status_code == 200
    assert response.json()["prediction"] in {"<=50K", ">50K"}


def test_docs_schema_has_hyphenated_names_and_examples():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()["components"]["schemas"]["CensusRecord"]
    properties = schema["properties"]
    # Hyphenated names are exposed through aliases ...
    for name in ("education-num", "marital-status", "capital-gain",
                 "capital-loss", "hours-per-week", "native-country"):
        assert name in properties
    # ... and every field, plus the body as a whole, documents an example.
    assert all("examples" in field for field in properties.values())
    assert schema["examples"][0]["education-num"] == 13


def test_record_to_features_matches_training_preprocessing():
    # The API must build exactly the feature matrix that training-time
    # processing builds for the same row: same column order, same encoder.
    # (Two clear-cut people would not reveal a scrambled column order.)
    data = pd.read_csv(DATA_PATH).sample(25, random_state=1)
    for _, row in data.iterrows():
        values = {key: (v.item() if hasattr(v, "item") else v)
                  for key, v in row.drop("salary").items()}
        record = CensusRecord.model_validate(values)
        expected, _, _, _ = process_data(
            pd.DataFrame([values], columns=FEATURE_COLUMNS),
            categorical_features=CAT_FEATURES,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        assert np.array_equal(record_to_features(record), expected)
