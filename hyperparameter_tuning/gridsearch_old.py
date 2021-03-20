from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sys, os
import pandas as pd
import json
import pprint
from preprocessing import create_log_folder
import mlflow
from globals import *
from evaluation import grouped_evaluation, get_scorer
from preprocessing import preprocessing
from utils import get_task, load_datasets
import click

# Default values
TASK = 'asset.variety'
IMPUTER = 'dropnan'
N_JOBS_CV = 5
N_FOLDS = 5
METRIC = 'auc'
AVERAGING = 'macro'
RANDOM_STATE = None


README = True
INDUSTRY_ONE_HOT = True

def mlflow_gridsearch(X, y, 
                      estimator_dict,
                      predictors,
                      cv, 
                      metric,
                      average, 
                      n_jobs, 
                      random_state):

    mlflow.sklearn.autolog()
    grs = GridSearchCV(estimator=estimator_dict['class'],
                       param_grid=estimator_dict['parameter_grid'],
                       cv=cv, 
                       scoring=get_scorer('all'), 
                       refit=metric,
                       n_jobs=n_jobs)
    with mlflow.start_run(run_name=f'Gridsearch - {metric}_{average}', nested=True) as run:
        mlflow.set_tags({'estimator_family': estimator_dict['family_name'], 
                         'target': y.name,
                         'predictors': predictors, 
                         'tuning_metric': f'{metric}_{average}'})
        grs.fit(X, y)
        grs.predict

    return grs

def ML_tuner_CV(X, y,  
                predictors,
                metric, 
                average,
                models=MODELS, 
                n_jobs_cv=-1, 
                n_folds=5,
                random_state=None): 

    """  ML classifier hyperparameter tuning """
    skf = StratifiedKFold(n_splits=n_folds, 
                          random_state=random_state)
    
    # loop on estimators 
    for estimator_dict in models:
        print(f'\n{estimator_dict["family_name"]}\n--------------------')
        # mlflow run on estimators
        with mlflow.start_run(run_name='estimator loop', nested=True) as model_run:
            mlflow.set_tags({'estimator_family': estimator_dict['family_name'], 
                             'target': y.name,
                             'predictors': predictors, 
                             'tuning_metric': f'{metric}_{average}'})
            mlflow_gridsearch(X, y, 
                              estimator_dict, 
                              predictors,
                              skf, 
                              metric,
                              average,
                              n_jobs_cv, 
                              random_state)

def grouped_search(X, y,
                   predictors, 
                   metric,
                   average,
                   n_jobs_cv=-1,
                   models=MODELS,
                   n_folds=5,
                   random_state=None,
                   readme=True):
    """ This function is a wrapper for gridsearch. It creates a loop for applying 
        hyperparameter tuning on multiple independent targets and helps to store 
        results a log inside a common mlflow experiment """  
    
    # Custom Logging
    if readme:
        full_features = {'Full features': X.columns.tolist()}
        collapsed_features = {'Collapsed features': predictors}
        targets = {'Targets': y.columns.tolist()}
        with open('full_feature_set.txt', 'w') as outfile:
            json.dump(full_features, outfile, indent=4)
        with open('simple_feature_set.txt', 'w') as outfile:
            json.dump(collapsed_features, outfile, indent=4)
        with open('targets.txt', 'w') as outfile:
            json.dump(targets, outfile, indent=4)
    
    # initial logs
    mlflow.start_run(run_name='parent run')


    mlflow.set_tags({'predictors':predictors, 
                     'tuning_metric': f'{metric}_{average}'})
    mlflow.log_artifact('full_feature_set.txt', artifact_path='features')
    mlflow.log_artifact('simple_feature_set.txt', artifact_path='features')
    mlflow.log_artifact('targets.txt', artifact_path='features')
    
    # loop over targets
    for target in y.columns:
        print(f'\n************************\n{target}\n************************')
        with mlflow.start_run(run_name='target loop', nested=True) as target_run:
            mlflow.set_tags({'target': target, 
                             'predictors': predictors, 
                             'tuning_metric': f'{metric}_{average}'})
            # Single target tuning
            ML_tuner_CV(X, y[target], 
                        predictors=predictors, 
                        metric=metric,
                        average=average,
                        n_jobs_cv=n_jobs_cv,
                        models=models,
                        n_folds=n_folds,
                        random_state=random_state)

@click.command()
@click.option("--task", "-t", 
              type=click.Choice(
                  ['asset.variety',
                   'asset.assets.variety',
                   'action',
                   'action.x.variety']),
              default=TASK,
              help="Learning task"
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan']),
              multiple=False, 
              default=IMPUTER,
              help="Imputation strategy"
              )
@click.option("--metric", "-m", 
              type=click.Choice(['auc', 'accuracy', 'precision', 'recall', 'f1']),
              default=METRIC, 
              help="Metric to maximise during tuning."
              )
@click.option("--averaging", "-a",
              type=click.Choice(['micro', 'macro', 'weighted']),
              default=AVERAGING,
              help="Method to compute aggregate metrics"
              )
@click.option("--n-folds", "-f", 
              type=int,
              default=N_FOLDS,
              help="Number of folds for CV"
              )
@click.option("--n-jobs-cv", "-j", 
              type=int,
              default=N_JOBS_CV,
              help="Number of cores for GridsearchCV"
              )
@click.option("--random-state", "-r", 
              type=int,
              default=RANDOM_STATE,
              help="Random state for GridsearchCV splits"
              )
def run(task, metric, averaging, imputer, n_folds, n_jobs_cv, random_state):
    # Render arguments 
    industry_one_hot = INDUSTRY_ONE_HOT
    readme = README
    random_state = None if random_state == -1 else random_state

    
    # Load data
    df, veris_df = load_datasets()
    
    # Manage task predictors and features
    predictors, targets = get_task(task, veris_df)

    # Preprocessing
    X, y = preprocessing(df, veris_df,
                         predictors, 
                         targets, 
                         industry_one_hot,
                         imputer)
            

    ## Hyperparameter tuning for training separate classifiers for each 2nd level action
    if metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            grouped_search(X, y, 
                           predictors=predictors, 
                           metric=metric,
                           average=averaging,
                           n_jobs_cv=n_jobs_cv,
                           models=MODELS, 
                           n_folds=n_folds,
                           random_state=random_state,
                           readme=readme)

if __name__ == '__main__':
    run()