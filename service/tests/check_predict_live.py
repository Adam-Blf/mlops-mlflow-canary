"""Manual smoke test of a running service's /predict endpoint.

Not collected by pytest (its name does not start with test_). Run it by
hand against a real container or a local uvicorn process:

    BASE_URL=http://localhost:8000 python tests/check_predict_live.py

It exits with a non-zero status and a clear message on any failure,
instead of a stack trace.
"""
import os
import sys

import requests

DEFAULT_BASE_URL = "http://localhost:8000"

SAMPLE_FEATURES = {
    "square_footage": 1800.0,
    "bedrooms": 4,
    "bathrooms": 2.5,
    "age_years": 15.0,
    "distance_to_city_km": 8.0,
}


def main() -> int:
    base_url = os.environ.get("BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=5)
    if health.status_code != 200:
        print(f"FAIL: /health returned {health.status_code}: {health.text}")
        return 1

    if not health.json().get("current_model_loaded"):
        print(f"FAIL: no model loaded yet, /health said: {health.json()}")
        return 1

    response = requests.post(
        f"{base_url}/predict", json={"features": SAMPLE_FEATURES}, timeout=10
    )
    if response.status_code != 200:
        print(f"FAIL: /predict returned {response.status_code}: {response.text}")
        return 1

    body = response.json()
    if "prediction" not in body or "served_by" not in body:
        print(f"FAIL: unexpected /predict response shape: {body}")
        return 1

    print(f"OK: /predict served by '{body['served_by']}' -> {body['prediction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
