# Training script

Trains a house price regressor on a synthetic, deterministic dataset
and logs the run to MLflow (params, metrics and the model itself,
registered under the name `house-price-regressor`).

## 1. Start the MLflow server

From the project root:

```
docker compose up -d mlflow
```

The UI is then reachable at http://localhost:5000.

## 2. Install the script dependencies

From this folder, in a virtual environment:

```
pip install -r requirements.txt
```

## 3. Run a training

```
python train.py
```

The script talks to MLflow through `MLFLOW_TRACKING_URI`, which
defaults to `http://localhost:5000` (host usage). Inside a container
on the same docker compose network, set it to `http://mlflow:5000`
instead.

## 4. Change a hyper parameter and compare runs

Every hyper parameter has a default and can be overridden, so two
runs coexist in the UI for comparison:

```
python train.py --n-estimators 200 --max-depth 10
python train.py --n-estimators 50 --max-depth 3 --run-name shallow-forest
```

Available options: `--n-estimators`, `--max-depth` (0 means
unbounded), `--min-samples-leaf`, `--n-samples`, `--test-size`,
`--run-name`.

Each run logs `mse` and `rmse` as metrics and registers a new
version of `house-price-regressor` in the Model Registry, so a
serving component can load a specific version by name.
