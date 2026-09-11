# Service

FastAPI service that serves a "house-price-regressor" model loaded from
an MLflow Model Registry at startup, with a current/next canary split.

## Endpoints

- `GET /health` - reports whether current and next models are loaded.
- `GET /model-status` - canary probability, version and request count per slot.
- `POST /predict` - body `{"features": {...}}`, routes to current with
  probability `p` and to next with probability `1 - p`, returns the
  prediction and which slot served it.
- `POST /update-model` - body `{"version": "<registry version or alias>"}`,
  loads that version into the `next` slot only.
- `POST /accept-next-model` - promotes `next` into `current`, both slots
  become identical again.

## Feature schema (`/predict`)

`square_footage`, `bedrooms`, `bathrooms`, `age_years`,

## Configuration (environment variables)

- `MLFLOW_TRACKING_URI` (default `http://localhost:5000`)
- `MLFLOW_MODEL_NAME` (default `house-price-regressor`)
- `MLFLOW_MODEL_VERSION` (default `latest`, used only at startup)
- `CANARY_PROBABILITY_CURRENT` (default `0.9`)
- `MODEL_LOAD_RETRY_ATTEMPTS` (default `5`)
- `MODEL_LOAD_RETRY_DELAY_SECONDS` (default `2.0`)

No model file is ever copied into the docker image, see the comment in
`Dockerfile`. The model is always fetched from MLflow at runtime.

## Running the tests

The suite under `tests/` runs fully offline, no MLflow server and no
docker needed, `load_model_with_retry` is monkeypatched with a stub
model.

```
pip install -r requirements.txt
pytest tests/
```

`tests/check_predict_live.py` is a separate manual smoke test that hits
a real running instance:

```
BASE_URL=http://localhost:8001 python tests/check_predict_live.py
```

## Running locally

```
pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn app.main:app --reload
```
