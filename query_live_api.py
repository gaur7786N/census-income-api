"""Send one POST request to the deployed API and print the result.

Usage::

    python query_live_api.py                       # uses LIVE_API_URL below
    python query_live_api.py https://my-app.onrender.com
    LIVE_API_URL=https://my-app.onrender.com python query_live_api.py

Set ``LIVE_API_URL`` below to the root URL of the deployed service once it
is live. The script prints the model's inference result and the HTTP
status code of the response.
"""

import os
import sys
from typing import Any, Dict, Optional, Sequence

import requests

# TODO: replace with the root URL of the deployed service (no trailing path).
LIVE_API_URL = "https://YOUR-APP-NAME.onrender.com"

PREDICT_PATH = "/predict"

# Free hosting tiers can take a while to wake an idle service.
TIMEOUT_SECONDS = 90

SAMPLE_PERSON: Dict[str, Any] = {
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


def resolve_base_url(argv: Sequence[str]) -> str:
    """Pick the base URL: command line, then environment, then constant."""
    if len(argv) > 1:
        return argv[1]
    return os.environ.get("LIVE_API_URL", LIVE_API_URL)


def query_api(base_url: str, person: Dict[str, Any]) -> requests.Response:
    """POST ``person`` to the prediction endpoint of ``base_url``."""
    return requests.post(
        base_url.rstrip("/") + PREDICT_PATH,
        json=person,
        timeout=TIMEOUT_SECONDS,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    base_url = resolve_base_url(argv)
    if "YOUR-APP-NAME" in base_url:
        print(
            "Set LIVE_API_URL in query_live_api.py (or pass the URL as an "
            "argument) to the URL of your deployed API."
        )
        return 2

    response = query_api(base_url, SAMPLE_PERSON)
    print(f"POST {base_url.rstrip('/')}{PREDICT_PATH}")
    print(f"HTTP status code: {response.status_code}")
    print(f"Model inference result: {response.json()}")
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
