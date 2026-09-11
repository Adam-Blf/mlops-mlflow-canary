# MLflow tracking server, built on top of the official image.
# curl is added only to support the docker compose healthcheck.

FROM ghcr.io/mlflow/mlflow:v2.17.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

EXPOSE 5000
