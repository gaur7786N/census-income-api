"""FastAPI application that serves the census salary classifier.

Run locally from the project root with::

    uvicorn main:app --reload

Interactive documentation, including example request values, is served at
``/docs``.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.config import (
    CAT_FEATURES,
    ENCODER_PATH,
    FEATURE_COLUMNS,
    LB_PATH,
    MODEL_PATH,
)
from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact

# Example request body shown in the generated documentation. The keys keep
# the hyphenated names used by the census data set.
EXAMPLE_RECORD: Dict[str, Any] = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


class CensusRecord(BaseModel):
    """One person's census attributes, as in a row of ``census.csv``.

    The data set has column names with hyphens, which are not valid Python
    identifiers. Each such field therefore has an alias, so the JSON body
    uses the original names (``education-num``) while the Python attribute
    uses an underscore (``education_num``).
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"examples": [EXAMPLE_RECORD]},
    )

    age: int = Field(..., ge=0, le=120, examples=[39])
    workclass: str = Field(..., min_length=1, examples=["State-gov"])
    fnlgt: int = Field(..., ge=0, examples=[77516])
    education: str = Field(..., min_length=1, examples=["Bachelors"])
    education_num: int = Field(
        ..., alias="education-num", ge=0, examples=[13]
    )
    marital_status: str = Field(
        ..., alias="marital-status", min_length=1, examples=["Never-married"]
    )
    occupation: str = Field(..., min_length=1, examples=["Adm-clerical"])
    relationship: str = Field(
        ..., min_length=1, examples=["Not-in-family"]
    )
    race: str = Field(..., min_length=1, examples=["White"])
    sex: str = Field(..., min_length=1, examples=["Male"])
    capital_gain: int = Field(
        ..., alias="capital-gain", ge=0, examples=[2174]
    )
    capital_loss: int = Field(
        ..., alias="capital-loss", ge=0, examples=[0]
    )
    hours_per_week: int = Field(
        ..., alias="hours-per-week", ge=0, le=168, examples=[40]
    )
    native_country: str = Field(
        ..., alias="native-country", min_length=1, examples=["United-States"]
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    prediction: str = Field(
        ..., description="Predicted salary class: '<=50K' or '>50K'.",
        examples=["<=50K"],
    )


app = FastAPI(
    title="Census Income Prediction API",
    description=(
        "Predicts whether a person earns more than 50K per year from "
        "US Census Bureau attributes."
    ),
    version="1.0.0",
)

# The artifacts are committed to the repository and are loaded once, when
# the application starts. Paths come from starter.config, which resolves
# them relative to the project root, so they work locally and when hosted.
model = load_artifact(MODEL_PATH)
encoder = load_artifact(ENCODER_PATH)
lb = load_artifact(LB_PATH)


@app.get("/")
def read_root() -> Dict[str, str]:
    """Greet the caller."""
    return {
        "message": (
            "Welcome to the Census Income Prediction API. "
            "Send a POST request to /predict to classify a person."
        )
    }


def record_to_features(record: CensusRecord) -> np.ndarray:
    """Turn one request body into the feature matrix the model expects.

    ``by_alias=True`` restores the original hyphenated column names, and
    passing ``columns`` fixes the column order to the one used in training.
    The fitted encoder from training is reused, never refitted.
    """
    frame = pd.DataFrame(
        [record.model_dump(by_alias=True)], columns=FEATURE_COLUMNS
    )
    X, _, _, _ = process_data(
        frame,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CensusRecord) -> PredictionResponse:
    """Predict whether the person earns '<=50K' or '>50K'."""
    X = record_to_features(record)
    predicted_class = lb.inverse_transform(inference(model, X))[0]
    return PredictionResponse(prediction=str(predicted_class))
