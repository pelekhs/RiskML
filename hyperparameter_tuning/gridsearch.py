from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sys, os
import pandas as pd
import json
import pprint
import mlflow
import click
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import logging


# please change in application -> run globals in the beginning and learn how to import from parent folder!!!!
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)
import models

from evaluation import get_scorer
from preprocessing import preprocessing
from utils import get_task, load_datasets, check_y_statistics
from utils import train_test_split_and_log

# Default values

default_arguments = {
    'task': 'asset.variety',
    'imputer': 'dropnan',
    'n_jobs_cv': 5,
    'n_folds': 5,
    'metric': 'auc',
    'averaging': 'macro',
    'random_state': 0,
    'train_size': 0.8,
    'industry_one_hot': True,
    'models': models.models
}

def mlflow_gridsearch(X, y, 
                      estimator_dict,
                      predictors,
                      n_folds, 
                      metric,
                      averaging, 
                      n_jobs):
    # Define gridsearch and autolog with mlflow
    mlflow.sklearn.autolog()
    # print(get_scorer('all'))
    grs = GridSearchCV(estimator=estimator_dict['class'](),
                       param_grid=estimator_dict['parameter_grid'],
                       cv=n_folds, 
                       scoring=get_scorer('gridsearch'), 
                       refit=metric,
                       n_jobs=n_jobs)
    # Start run and log
    family = estimator_dict['family_name']
    with mlflow.start_run(run_name=f'gridsearch: {family}', nested=True) as run:
        mlflow.set_tags({'estimator_family': family, 
                         'target': y.name,
                         'predictors': predictors, 
                         'tuning_metric': f'{metric}_{averaging}'})
        try:
            grs.fit(X, y.values.reshape(-1))
        except ValueError:
            mlflow.set_tags({'failed': '1 class only'})
            return
    return grs

def ML_tuner_CV(X, y,  
                predictors,
                metric, 
                averaging,
                models=default_arguments['models'], 
                n_jobs_cv=-1, 
                n_folds=5): 

    """  ML classifier hyperparameter tuning """
    
    # loop on estimators 
    for estimator_dict in models:
        print(f'\n{estimator_dict["family_name"]}\n--------------------')
        # mlflow run on estimators
        estimator_name = estimator_dict['family_name']
        mlflow_gridsearch(X, y, 
                          estimator_dict, 
                          predictors,
                          n_folds, 
                          metric,
                          averaging,
                          n_jobs_cv)

@click.command()
@click.option("--task", "-t", 
              type=click.Choice(
                  ['asset.variety',
                   'asset.assets.variety.S',
                   'asset.assets.variety.M',
                   'asset.assets.variety.U',
                   'asset.assets.variety.P',
                   'asset.assets.variety.T',
                   'action',
                   'action.error.variety',
                   'action.hacking.variety',
                   'action.misuse.variety',
                   'action.physical.variety',
                   'action.malware.variety',
                   'action.social.variety']),
              default=default_arguments['task'],
              help="Learning task"
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan']),
              multiple=False, 
              default=default_arguments['imputer'],
              help="Imputation strategy"
              )
@click.option("--metric", "-m", 
              type=click.Choice(['auc', 'accuracy', 'precision', 'recall', 'f1', 'hl']),
              default=default_arguments['metric'], 
              help="Metric to maximise during tuning."
              )
@click.option("--averaging", "-a",
              type=click.Choice(['micro', 'macro', 'weighted']),
              default=default_arguments['averaging'],
              help="Method to compute aggregate metrics"
              )
@click.option("--n-folds", "-f", 
              type=int,
              default=default_arguments['n_folds'],
              help="Number of folds for CV"
              )
@click.option("--n-jobs-cv", "-j", 
              type=int,
              default=default_arguments['n_jobs_cv'],
              help="Number of cores for GridsearchCV"
              )
@click.option("--random-state", "-r", 
              type=int,
              default=default_arguments['random_state'],
              help="Random state for train/test splits"
              )
@click.option("--train-size", "-t", 
              type=float,
              default=default_arguments['train_size'],
              help="Training set percentage of the dataset size"
              )

def run(task, metric, averaging, imputer, n_folds, n_jobs_cv, random_state, train_size):

    # Load data
    veris_df = load_dataset(collapsed=collapsed_csv_name)
    
    # Manage task predictors and features
    predictors, targets = get_task(task, veris_df)

    print("Targets:", targets)
    
    # Custom Logs
    collapsed_features = {'Collapsed features': predictors}

    subtasks = {'Subtasks': targets}
    
    with open('simple_feature_set.txt', 'w') as f:
    
        json.dump(collapsed_features, f, indent=4)
    
    with open('targets.txt', 'w') as f:
    
        json.dump(subtasks, f, indent=4)
    
    metric_id = ("-".join[metric, averaging]) if (metric not in ['hl', 'accuracy']) else metric
    
    for target in targets:
    
        print(f'\n************************\n{target}\n************************')
    
        with mlflow.start_run(run_name=f'{target} | {metric}-{averaging}'):
    
            # Preprocessing
            X, y = preprocessing(df, veris_df,
                                    predictors, 
                                    target, 
                                    default_arguments['industry_one_hot'],
                                    imputer)
            # Log tags and params
            mlflow.set_tag("mlflow.runName", f'{target} | {metric}-{averaging}')
    
            mlflow.log_artifact('simple_feature_set.txt', 
                                artifact_path='features')
    
            mlflow.log_artifact('targets.txt', 
                                artifact_path='targets')

            mlflow.set_tags({'target': target, 
                                'predictors': predictors,
                                'n_samples': len(y),
                                'class_balance': sum(y)/len(y),
                                'tuning_metric': metric_id,
                                'imputer': imputer,
                                'n_folds': n_folds,
                                'random_state': random_state})
    
            mlflow.log_params({'tuning_metric': metric_id,
                                'imputer': imputer,
                                'n_folds': n_folds,
                                'random_state': random_state})
    
            # check y statistics and log
            error_tags = check_y_statistics(y)
    
            mlflow.set_tags(error_tags)

            # skip loop if y is small
            if mlflow_tags['error_class'] != '-':
    
                continue

            # train / test if train percentage < 1
            X_train, X_test, y_train, y_test = \
                train_test_split_and_log(X, y, train_size, random_state)

            # Single target tuning only on train data.
            ML_tuner_CV(X_train, y_train, 
                        predictors=predictors, 
                        metric=metric,
                        averaging=averaging,
                        n_jobs_cv=n_jobs_cv,
                        models=default_arguments['models'],
                        n_folds=n_folds)

if __name__ == '__main__':
    run()