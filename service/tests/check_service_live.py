"""End to end check of a running service, the full canary cycle.

The unit tests run against a stub model, so they prove the routing logic
and nothing about the real contract with MLflow. This script proves the
rest: that a model really loads from the registry, that a prediction
really comes back, that /update-model really changes the next slot only,
that traffic really splits, and that /accept-next-model really promotes.

Not collected by pytest, its name does not start with test_. Run it
against a running stack:

    BASE_URL=http://localhost:8001 python tests/check_service_live.py

Exits non zero on the first failed step, with a readable message rather
than a stack trace.
"""
import os
import sys
from collections import Counter

import requests

DEFAULT_BASE_URL = "http://localhost:8001"
TIMEOUT = 10
N_CANARY_REQUESTS = 60

# Types matter here: the model signature declares integers for bedrooms,
# bathrooms and age_years, and MLflow refuses to convert float64 to int64
# at predict time. Sending 2.5 bathrooms is rejected before the model.
SAMPLE_FEATURES = {
    "square_footage": 1800.0,
    "bedrooms": 4,
    "bathrooms": 2,
    "age_years": 15,
    "distance_to_city_km": 8.0,
}


class CheckFailed(Exception):
    """Raised with the message to print, so main stays flat."""


def etape(numero: int, titre: str) -> None:
    print(f"\n[{numero}] {titre}")


def get_json(url: str) -> dict:
    reponse = requests.get(url, timeout=TIMEOUT)
    if reponse.status_code != 200:
        raise CheckFailed(f"GET {url} returned {reponse.status_code}: {reponse.text}")
    return reponse.json()


def post_json(url: str, payload: dict | None = None) -> dict:
    reponse = requests.post(url, json=payload, timeout=TIMEOUT)
    if reponse.status_code != 200:
        raise CheckFailed(f"POST {url} returned {reponse.status_code}: {reponse.text}")
    return reponse.json()


def verifier(base_url: str) -> None:
    etape(1, "health, a model must be loaded in both slots")
    sante = get_json(f"{base_url}/health")
    if sante.get("status") != "ok":
        raise CheckFailed(f"service is not healthy: {sante}")
    print(f"    ok, {sante['detail']}")

    etape(2, "model-status, both slots identical at startup")
    statut = get_json(f"{base_url}/model-status")
    version_depart = statut["current"]["version"]
    print(f"    current=v{version_depart}, next=v{statut['next']['version']}, "
          f"p={statut['canary_probability_current']}")

    etape(3, "predict, a real prediction comes back")
    prediction = post_json(f"{base_url}/predict", {"features": SAMPLE_FEATURES})
    for champ in ("prediction", "served_by", "model_version"):
        if champ not in prediction:
            raise CheckFailed(f"missing '{champ}' in /predict response: {prediction}")
    print(f"    ok, {prediction['prediction']:.2f} served by "
          f"'{prediction['served_by']}' on v{prediction['model_version']}")

    etape(4, "update-model, only the next slot changes")
    cible = "1" if version_depart != "1" else "2"
    mise_a_jour = post_json(f"{base_url}/update-model", {"version": cible})
    if mise_a_jour.get("slot") != "next":
        raise CheckFailed(f"/update-model touched the wrong slot: {mise_a_jour}")

    apres = get_json(f"{base_url}/model-status")
    if apres["current"]["version"] != version_depart:
        raise CheckFailed(
            "current slot moved, it must not: "
            f"{version_depart} -> {apres['current']['version']}"
        )
    if apres["next"]["version"] != cible:
        raise CheckFailed(f"next slot did not move to v{cible}: {apres}")
    print(f"    ok, current stayed on v{version_depart}, next is now v{cible}")

    etape(5, f"predict x{N_CANARY_REQUESTS}, traffic really splits")
    comptes: Counter = Counter()
    for _ in range(N_CANARY_REQUESTS):
        reponse = post_json(f"{base_url}/predict", {"features": SAMPLE_FEATURES})
        comptes[reponse["served_by"]] += 1

    if comptes["next"] == 0:
        raise CheckFailed(
            f"no request reached the canary in {N_CANARY_REQUESTS} calls, "
            "routing looks stuck on current"
        )
    if comptes["current"] == 0:
        raise CheckFailed("no request reached current, routing looks inverted")

    attendu = apres["canary_probability_current"]
    mesure = comptes["current"] / N_CANARY_REQUESTS
    print(f"    ok, current={comptes['current']} next={comptes['next']}, "
          f"measured p={mesure:.2f} against {attendu} configured")

    etape(6, "accept-next-model, the canary becomes current")
    promotion = post_json(f"{base_url}/accept-next-model")
    final = get_json(f"{base_url}/model-status")
    if final["current"]["version"] != cible:
        raise CheckFailed(f"promotion did not take: {final}")
    print(f"    ok, current is now v{final['current']['version']}")

    etape(7, "predict after promotion, served by the promoted version")
    derniere = post_json(f"{base_url}/predict", {"features": SAMPLE_FEATURES})
    if derniere["model_version"] != cible:
        raise CheckFailed(
            f"prediction still served by v{derniere['model_version']}, "
            f"expected v{cible}"
        )
    print(f"    ok, {derniere['prediction']:.2f} on v{derniere['model_version']}")

    etape(8, "bad input is rejected before the model")
    mauvais = dict(SAMPLE_FEATURES, bathrooms=2.5)
    reponse = requests.post(
        f"{base_url}/predict", json={"features": mauvais}, timeout=TIMEOUT
    )
    if reponse.status_code != 422:
        raise CheckFailed(
            "a float where the model signature wants an integer should be "
            f"rejected with 422, got {reponse.status_code}: {reponse.text}"
        )
    print("    ok, 422 as expected")


def main() -> int:
    base_url = os.environ.get("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    print(f"checking {base_url}")
    try:
        verifier(base_url)
    except CheckFailed as echec:
        print(f"\nFAIL: {echec}")
        return 1
    except requests.RequestException as echec:
        print(f"\nFAIL: service unreachable at {base_url}: {echec}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
