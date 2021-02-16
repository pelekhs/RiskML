FROM python:3.8-slim

RUN pip install --upgrade pip && \
    pip install numpy argparse docker pandas scikit-learn matplotlib lightgbm mlflow