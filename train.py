from evaluation import get_scorer
from mlflow import log_metric, log_param, log_artifacts
from globals import MODELS

def train(X, y, estimator, train_size, metrics, metrics_averaging, random_state)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, 
                         train_size=train_size,
                         test_size=1-train_size,
                         shuffle=True,
                         stratify=y,
                         random_state=random_state)
    estimator = MODELS[estimator]
    estimator.fit(X_train, y_train)
    # Evaluate for the given metrics
    for metric, averaging in list(zip(metrics, metrics_averaging)):
        scorer = get_scorer(metric, averaging)
        score = scorer(estimator, X_test, y_test)
        mlflow.log_param()
        mlflow.log_params(estimator.get_params())
        mlflow.log_metric(" ".join(metric, averaging), score)