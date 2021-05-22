#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI='postgresql+psycopg2://postgres:lollipops@localhost:5432/postgres'

# Serve the production model from the model registry
mlflow models serve -m "models:/asset.variety.Server/Production"

