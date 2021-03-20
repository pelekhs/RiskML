#%%
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import grouped_tuning
from evaluation import grouped_evaluation
from preprocessing import preprocessor, ColToOneHot, load_datasets
import argparse
import click
from globals import *
from preprocessing import preprocessing

# Default Arguments
## General
PREDICTORS = ["action", "action.x.variety", "victim.industry", "victim.orgsize"]
TARGETS = ["asset.variety.Server"]#"asset.variety.Media", "asset.variety.User Dev"]
TRAIN_SIZE = 0.8
INDUSTRY_ONE_HOT = True
IMPUTER = 'dropnan'
## Tuning
TUNING_METRIC = "auc"
TUNING_AVERAGING = "macro"
N_JOBS_CV = -1
FOLDER_NAME = "asset"
README = True

@click.command()
@click.option("--train-size", "-tr", 
              type=float, 
              default=TRAIN_SIZE,
              help="training set size"
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
              multiple=True,
              default=TARGETS,
              help="Independent targets of the model"
              )
@click.option("--tuning-metric", "-tm", 
              type=click.Choice(['auc', 'accuracy', 'precision', 'recall', 'f1']),
              default=TUNING_METRIC, 
              help="Metric to maximise during tuning."
              )
@click.option("--tuning-averaging", "-ta",
              type=click.Choice(['micro', 'macro', 'weighted']),
              default=TUNING_AVERAGING,
              help="Method to compute aggregate metrics"
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan']),
              default=IMPUTER,
              help="Independent targets of the model"
              )
# @click.option("--stratify", 
#               type=bool, 
#               default=STRATIFY, 
#               help="Target according to which the train/test split is stratified"
#               )
def ML_asset(predictors, targets, train_size, 
             tuning_metric, tuning_averaging,
             imputer):

    # Render arguments 
    targets=TARGETS if targets[0] == ' '  else targets
    predictors=PREDICTORS if predictors[0] == ' ' else predictors
    
    # Preprocessing
    X, ys = preprocessing(predictors, targets, 
                          INDUSTRY_ONE_HOT,
                          imputer)

    ## Train / Test split
    stratifier = targets[0]
    X_train, X_test, y_trains, y_tests = \
        train_test_split(X, ys, train_size=train_size,
                         test_size=1-train_size,
                         shuffle=True,
                         stratify=ys[stratifier])
    ## Hyperparameter Tuning for training separate classifiers for each 2nd level action
    if tuning_metric in ["auc", "accuracy", "precision", "recall", "f1"]:
        best_scores, best_models, best_params = \
            grouped_tuning(X_train, X_test, y_trains, y_tests,
                           results_dir=RESULTS_DIR, 
                           pipeline=FOLDER_NAME,
                           tune_metric=tuning_metric, 
                           average=tuning_averaging,
                           models=MODELS,
                           param_grid=PARAM_GRID,
                           n_jobs_cv=N_JOBS_CV,
                           readme=README)
    
    ## Evaluation
    ### Evaluation should take best models from above and
    ### include all metrics! After MLflow do that
    eval_scores = grouped_evaluation(X_train, X_test, 
                                     y_trains, y_tests,
                                     models=MODELS)

if __name__ == "__main__":
    ML_asset()

    
# %%
