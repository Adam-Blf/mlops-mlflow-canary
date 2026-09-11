"""Train a house price regressor and log it to MLflow.

The dataset is synthetic and generated deterministically with a fixed
numpy seed, so the script never depends on the network. Hyper
parameters are exposed on the command line so a second run with a
different value can be compared to the first one in the MLflow UI.

The trained model is logged with mlflow.sklearn.log_model and
registered under a fixed name in the Model Registry, so a serving
component can later load it by name and version.
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "house-price-regression"
REGISTERED_MODEL_NAME = "house-price-regressor"
DATASET_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a house price regressor")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="number of trees in the random forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="maximum depth of each tree, 0 means unbounded (default: 6)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="minimum number of samples required at a leaf node (default: 1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="number of synthetic houses to generate (default: 2000)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="fraction of the dataset used for the test split (default: 0.2)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional name for the MLflow run",
    )
    return parser.parse_args()


def make_house_price_dataset(n_samples: int, seed: int = DATASET_SEED) -> pd.DataFrame:
    """Generate a deterministic synthetic house price dataset.

    Features roughly mimic a real estate dataset: square footage,
    number of bedrooms and bathrooms, age of the house and distance
    to the city center. The target price is a linear combination of
    these features plus gaussian noise, so a tree based model has a
    real relationship to learn.
    """
    rng = np.random.default_rng(seed)

    square_footage = rng.normal(loc=1800, scale=650, size=n_samples).clip(min=300)
    bedrooms = rng.integers(low=1, high=6, size=n_samples)
    bathrooms = rng.integers(low=1, high=4, size=n_samples)
    age_years = rng.integers(low=0, high=80, size=n_samples)
    distance_to_city_km = rng.exponential(scale=12, size=n_samples).clip(max=90)

    noise = rng.normal(loc=0, scale=15000, size=n_samples)

    price = (
        120 * square_footage
        + 8000 * bedrooms
        + 6000 * bathrooms
        - 800 * age_years
        - 1500 * distance_to_city_km
        + 50000
        + noise
    ).clip(min=20000)

    return pd.DataFrame(
        {
            "square_footage": square_footage,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "distance_to_city_km": distance_to_city_km,
            "price": price,
        }
    )


def main() -> None:
    args = parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = make_house_price_dataset(args.n_samples)
    feature_columns = [
        "square_footage",
        "bedrooms",
        "bathrooms",
        "age_years",
        "distance_to_city_km",
    ]
    x = dataset[feature_columns]
    y = dataset["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=DATASET_SEED
    )

    max_depth = None if args.max_depth == 0 else args.max_depth

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("n_samples", args.n_samples)
        mlflow.log_param("test_size", args.test_size)

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=DATASET_SEED,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=x_train.head(5),
        )

        print(f"mse: {mse:.2f}")
        print(f"rmse: {rmse:.2f}")
        print(f"registered model: {REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
