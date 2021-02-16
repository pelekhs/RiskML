#%%
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import grouped_tuning
from evaluation import grouped_evaluation
from preprocessing import preprocessor, ColToOneHot, load_datasets
import argparse
import click
from globals import *
from preprocessing import preprocessing
from sklearn.multioutput import MultiOutputClassifier
#from train import train
import mlflow

# Default Arguments
## General
PREDICTORS = ["action", "action.x.variety", "victim.industry", "victim.orgsize"]
TARGET = ["asset.variety.Server"]#"asset.variety.Media", "asset.variety.User Dev"]
TRAIN_SIZE = 0.8
INDUSTRY_ONE_HOT = True
IMPUTER = 'dropnan'
## Tuning
TUNING_METRIC = "auc"
TUNING_AVERAGING = "macro"
N_JOBS_CV = -1
FOLDER_NAME = "asset"
README = True
#%%

@click.command()
@click.option("--train-size", "-tr", 
              type=float, 
              default=TRAIN_SIZE,
              help="training set size"
              )
@click.option("--random-state", "-rs", 
              type=int, 
              default=None,
              help="Random state for splitting train / test"
              )
@click.option("--estimator", "-e", 
              type=click.Choice(['SVM',
                                'KNN', 
                                'RF', 
                                'LR',
                                'LGBM',
                                'GNB'])
              default="SVM",
              help="ML algorithm to train models"
              )
@click.option("--predictors", "-p", 
              type=click.Choice([' ',
                                'action', 
                                'action.x.variety', 
                                'asset.variety',
                                'asset.assets.variety',
                                'victim.industry', 
                                'victim.orgsize']),
              multiple=True, 
              default=PREDICTORS,
              help="Predictors of the model"
              )
@click.option("--targets", "-t", 
              type=click.Choice(
                  [' ',
                  'asset.variety.Embedded', 
                  'asset.variety.Kiosk/Term',
                  'asset.variety.Media', 
                  'asset.variety.Network',
                  'asset.variety.Person', 
                  'asset.variety.Server',
                  'asset.variety.User Dev']),
              multiple=False,
              default=TARGETS,
              help="Independent targets of the model"
              )
@click.option("-metrics", "m", 
              type=click.Choice(['auc', 'accuracy', 'precision', 'recall', 'f1']),
              default=TUNING_METRIC, 
              multiple=True,
              help="Metric to maximise during tuning."
              )
@click.option("--metrics-averaging", "-ma",
              type=click.Choice(['micro', 'macro', 'weighted']),
              default=TUNING_AVERAGING,
              multiple=True,
              help="Method to compute aggregate metrics"
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan']),
              default=IMPUTER,
              help="Independent targets of the model"
              )
#%%
def run(predictors, targets, train_size, 
        metric, metric_averaging,
        imputer):
    mlflow.set_experiment(f"asset_{targets}")
    # Render arguments 
    targets = TARGETS if targets[0] == ' '  else targets
    predictors = PREDICTORS if predictors[0] == ' ' else predictors
    random_state = None if random_state == -1 else random_state
    
    # Preprocessing
    X, y = preprocessing(predictors, targets, 
                         INDUSTRY_ONE_HOT,
                         imputer)

    ## Train
    with mlflow.start_run():
        mlflow.log_param("predictors", predictors)

        train(X=X, y=y, 
            estimator=MODELS[estimator],
            train_size=train_size
            metric=metric, averaging=metric_averaging,
            random_state=random_state)  

# """Move to new file -- entrypoint """
#     ## Hyperparameter Tuning for training separate classifiers for each 2nd level action
#     if tuning_metric in ["auc", "accuracy", "precision", "recall", "f1"]:
#         best_scores, best_models, best_params = \
#             grouped_tuning(X_train, X_test, y_trains, y_tests,
#                            results_dir=RESULTS_DIR, 
#                            pipeline=FOLDER_NAME,
#                            tune_metric=tuning_metric, 
#                            average=tuning_averaging,
#                            models=MODELS,
#                            param_grid=PARAM_GRID,
#                            n_jobs_cv=N_JOBS_CV,
#                            readme=README)
    
#  """Should leave from here and be integrated in train"""   
#     ## Evaluation
#     ### Evaluation should take best models from above and
#     ### include all metrics! After MLflow do that
#     eval_scores = grouped_evaluation(X_train, X_test, 
#                                      y_trains, y_tests,
#                                      models=MODELS)

if __name__ == "__main__":
    run()

    
# %%
