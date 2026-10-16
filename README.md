# Census Income Prediction API

A scikit-learn classifier that predicts whether a person earns more than
50K per year from US Census Bureau data, served by a FastAPI application with
tests, continuous integration and continuous deployment.

| | |
|---|---|
| **Repository platform** | GitHub (CI and CD with GitHub Actions, hosting on Render) |
| **Repository URL** | TODO: https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY-NAME |
| **Live API URL** | TODO: https://YOUR-APP-NAME.onrender.com |
| **Python** | 3.13 (development, CI and Render all use 3.13) |

> Replace the two TODO values above before submitting.

## Project layout

```
main.py                    FastAPI app (GET /, POST /predict)
query_live_api.py          requests script that POSTs to the live API
starter/
  config.py                feature names, label, artifact and data paths
  train_model.py           training script: python -m starter.train_model
  ml/data.py               process_data (one hot encoding + label binarizer)
  ml/model.py              train, inference, metrics, save/load, slice metrics
model/                     model.joblib, encoder.joblib, lb.joblib (committed)
data/census.csv            cleaned data (spaces removed, column names unchanged)
slice_output.txt           model metrics for every value of each categorical feature
model_card.md              model card
tests/                     pytest suite (model, training script, API, client script)
sanitycheck.py             course helper that scans the API tests
.github/workflows/ci.yml   flake8 + pytest, then deployment
screenshots/               required screenshots
```

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run every command below from the repository root.

## Data

`data/census.csv` is the UCI Census Income ("Adult") data. The original file
has a space after every comma, which makes pandas read column names such as
`" workclass"` and values such as `" State-gov"`. The spaces were removed from
the file; the column names (including the hyphenated ones) were not changed.

## Model

```bash
python -m starter.train_model
```

The script loads and processes the data, splits it 80/20 (stratified,
`random_state=42`), trains a `RandomForestClassifier`, prints the test-set
metrics and the slice metrics, saves the model and the fitted encoder and
label binarizer to `model/`, and writes the slice metrics to
`slice_output.txt`.

The trained artifacts are committed so that a fresh clone can run inference
without retraining. They are ignored by the starter `.gitignore`, so they were
added explicitly:

```bash
git add -f model/model.joblib model/encoder.joblib model/lb.joblib
```

Load them with `joblib` (via `starter.ml.model.load_artifact`). They depend on
the pinned `scikit-learn`, `numpy` and `pandas` versions in
`requirements.txt`.

Held-out test performance (precision / recall / F1): 0.7847 / 0.6135 / 0.6886.
See `model_card.md` for details and `slice_output.txt` for slice metrics.

## API

```bash
uvicorn main:app --reload
```

* `GET /` returns a welcome message.
* `POST /predict` takes a JSON body with the 14 census fields and returns
  `{"prediction": "<=50K"}` or `{"prediction": ">50K"}`.
* Interactive documentation with example values is at `/docs`.

The census columns contain hyphens (for example `education-num`), which are
not valid Python names. The Pydantic model keeps the CSV's names as field
aliases, so the JSON body uses `education-num` while the Python attribute is
`education_num`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "workclass": "Private", "fnlgt": 160187,
       "education": "Masters", "education-num": 14,
       "marital-status": "Married-civ-spouse",
       "occupation": "Exec-managerial", "relationship": "Husband",
       "race": "White", "sex": "Male", "capital-gain": 15024,
       "capital-loss": 0, "hours-per-week": 50,
       "native-country": "United-States"}'
```

## Tests and linting

```bash
flake8 .
pytest
echo tests/test_api.py | python sanitycheck.py
```

The test suite covers the data and model functions, the training script, the
API (one GET test, one POST test for each of the two possible predictions,
and validation tests) and the live-API client script.

## Continuous integration and deployment

`.github/workflows/ci.yml` runs on every push to `main` or `master` and on pull
requests into them:

1. **test** job: Python 3.13, `pip install -r requirements.txt`,
   `flake8 .`, then `pytest`. Both must pass.
2. **deploy** job: runs only for a push to `main` or `master`, and only
   `needs: test`, so a failing flake8 or pytest run never deploys. It calls
   the Render deploy hook stored in the `RENDER_DEPLOY_HOOK_URL` repository
   secret. No credentials are stored in the repository.

### Render setup

1. Create a Render **Web Service** from this GitHub repository (branch `main`).
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Python version: the `.python-version` file pins 3.13 (Render's default for
   new services is newer, so keep this file).
5. Set **Auto-Deploy** to **Off** in Render so that deployments are triggered
   only by the GitHub Actions deploy job after CI passes.
6. Copy the service's **Deploy Hook** URL and save it as the GitHub repository
   secret `RENDER_DEPLOY_HOOK_URL` (Settings > Secrets and variables >
   Actions). Treat it like a password: do not commit or screenshot it.
7. Protect the `main` branch (Settings > Branches) and require the `test`
   status check.

The deploy job prints a warning and skips deployment if the secret is not set.

### Query the live API

Set `LIVE_API_URL` in `query_live_api.py` (or pass the URL as an argument) and
run:

```bash
python query_live_api.py https://YOUR-APP-NAME.onrender.com
```

It sends one POST request with `requests` and prints the HTTP status code and
the model's inference result.

## Required screenshots (in `screenshots/`)

| File | Shows |
|---|---|
| `continuous_integration.png` | Passing GitHub Actions run (flake8 and pytest) |
| `example.png` | FastAPI `/docs` page showing the example request values |
| `continuous_deployment.png` | Deploy job succeeded / continuous deployment enabled |
| `live_get.png` | Browser showing the GET response from the live root URL |
| `live_post.png` | Output of `query_live_api.py` against the live API |

Images in `screenshots/` are ignored by `.gitignore`, so each one is added
explicitly with `git add -f screenshots/<file>.png`.
