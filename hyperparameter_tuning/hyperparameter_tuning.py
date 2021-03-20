from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sys, os
import pandas as pd
import json
import pprint
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from preprocessing import create_log_folder
import mlflow
from globals import *
from sklearn.metrics import recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix, roc_auc_score
from evaluation import grouped_evaluation
from preprocessing import preprocessing, load_datasets
from utils import get_task

# Default Arguments
## Preprocessing
#PREDICTORS = ['action', 'action.x.variety', 'victim.industry', 'victim.orgsize']
#TARGETS = ['asset.variety.Server', 'asset.variety.User Dev']#'asset.variety.Media',]
TASK_NAME = "asset.variety"
INDUSTRY_ONE_HOT = True
IMPUTER = 'dropnan'
## Tuning
METRIC = 'accuracy'
AVERAGING = 'macro'
N_JOBS_CV = 4
README = True
N_FOLDS = 5
RANDOM_STATE=None # random state for cv

def get_scorer(metric, average):
    scorers = {
              'precision': make_scorer(precision_score, average=average),
              'recall': make_scorer(recall_score, average=average),
              'accuracy': make_scorer(accuracy_score),
              'f1': make_scorer(f1_score, average=average),
              'auc': make_scorer(roc_auc_score, average=average)
              }
    return scorers[metric]

def mlflow_gridsearch(X, y, 
                      estimator_dict,
                      predictors,
                      cv, 
                      metric,
                      average, 
                      n_jobs, 
                      random_state):

    mlflow.sklearn.autolog()
    clf = GridSearchCV(estimator=estimator_dict['class'],
                       param_grid=estimator_dict['parameter_grid'],
                       cv=cv, 
                       scoring=get_scorer(metric, average), 
                       refit=True,
                       n_jobs=n_jobs)
    with mlflow.start_run(run_name=f'gridsearch on {metric}', nested=True) as run:
        mlflow.set_tags({'estimator_family': estimator_dict['family_name'], 
                         'target': y.name,
                         'predictors': predictors, 
                         'tuning_metric': f'{metric}_{average}'})
        clf.fit(X, y)
        clf.predict
        print(clf.cv_results_)

    return clf

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
            clf = mlflow_gridsearch(X, y, 
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
                   experiment_id=None,
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
    mlflow.start_run(experiment_id=experiment_id, 
                     run_name='parent run')
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

if __name__ == '__main__':

    # Parse arguments (replace with click() and run())
    task_name = TASK_NAME
    industry_one_hot = INDUSTRY_ONE_HOT
    imputer =  IMPUTER
    metric = METRIC
    averaging = AVERAGING
    n_jobs_cv = N_JOBS_CV
    n_folds = N_FOLDS
    random_state=RANDOM_STATE
    readme = README

    # Set experiment
    experiment_id = mlflow.set_experiment(task_name)
    print(experiment_id)
    #experiment_id = mlflow.set_experiment(TARGETS[0].split('.')[0])
    
    # Load data
    df, veris_df = load_datasets()
    
    # Manage task predictors and features
    predictors, targets = get_task(task_name, veris_df)

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
                           readme=readme,
                           experiment_id=experiment_id)