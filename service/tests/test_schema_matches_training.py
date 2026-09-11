"""The API schema must match the trained model, names AND dtypes.

This guard exists because of a real failure. The unit tests below run
against a stub model that accepts anything, so they stayed green while
every real prediction returned a 500: MLflow enforces the model
signature at predict time and refuses to convert float64 to int64, and
the API declared bathrooms and age_years as float while the training
script produced integers.

A stub model can never catch that. Comparing the Pydantic field types
against the dtypes the training script actually produces can, and it
runs offline, without MLflow and without docker.
"""
import pathlib
import sys

import pandas as pd
import pytest

RACINE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "scripts"))

from train import make_house_price_dataset  # noqa: E402

from app.schemas import HouseFeatures  # noqa: E402

TARGET = "price"


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    return make_house_price_dataset(n_samples=50)


def test_feature_names_match_training(training_frame: pd.DataFrame) -> None:
    attendues = set(training_frame.columns) - {TARGET}
    declarees = set(HouseFeatures.model_fields)
    assert declarees == attendues, (
        "the API feature names drifted from the training script: "
        f"missing {attendues - declarees}, unexpected {declarees - attendues}"
    )


def test_feature_types_match_training(training_frame: pd.DataFrame) -> None:
    ecarts = []
    for nom, champ in HouseFeatures.model_fields.items():
        dtype = training_frame[nom].dtype
        entier_cote_modele = pd.api.types.is_integer_dtype(dtype)
        entier_cote_api = champ.annotation is int

        if entier_cote_modele != entier_cote_api:
            ecarts.append(
                f"{nom}: training produces {dtype}, API declares "
                f"{champ.annotation.__name__}"
            )

    assert not ecarts, (
        "MLflow signature enforcement will reject these at predict time, "
        "float64 cannot be safely converted to int64: " + "; ".join(ecarts)
    )
