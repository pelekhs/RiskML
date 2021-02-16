from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sys, os
import pandas as pd
import json
import pprint
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from preprocessing import create_log_folder
from sklearn.svm import SVC
import mlflow
from globals import *
from sklearn.metrics import recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from evaluation import grouped_evaluation
from globals import *
from preprocessing import preprocessing, load_datasets

# Default Arguments
## Preprocessing
PREDICTORS = ['action', 'action.x.variety', 'victim.industry', 'victim.orgsize']
TARGETS = ['asset.variety.Server', 'asset.variety.User Dev']#'asset.variety.Media',]
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
    with mlflow.start_run(run_name='gridsearch', nested=True) as run:
        mlflow.set_tags({'estimator_family': estimator_dict['family_name'], 
                         'target': y.name,
                         'predictors': predictors, 
                         'tuning_metric': f'{metric}_{average}'})
        clf.fit(X, y)
    return clf

def ML_tuner_CV(X, y,  
                models=MODELS, 
                predictors=PREDICTORS,
                metric=METRIC, 
                average=AVERAGING,
                n_jobs_cv=N_JOBS_CV, 
                n_folds=N_FOLDS,
                random_state=RANDOM_STATE): 

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
                   metric=METRIC,
                   average=AVERAGING,
                   n_jobs_cv=N_JOBS_CV,
                   models=MODELS,
                   predictors=PREDICTORS,
                   n_folds=N_FOLDS,
                   random_state=RANDOM_STATE,
                   experiment_id=None,
                   readme=True):
    
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
                        metric=metric,
                        average=average,
                        n_jobs_cv=n_jobs_cv,
                        models=models,
                        predictors=predictors, 
                        n_folds=n_folds,
                        random_state=random_state)

if __name__ == '__main__':

    experiment_id = mlflow.set_experiment("skata")
    #experiment_id = mlflow.set_experiment(TARGETS[0].split('.')[0])
    
    # Load data
    df, veris_df = load_datasets()
    
    # Manage task predictors and features
    if task == "asset.variety":
        predictors = ['action', 
                      'action.x.variety', 
                      'victim.industry', 
                      'victim.orgsize']
        targets = veris_df.filter(like="asset.variety.").columns.tolist()
        targets.remove("asset.variety.Embedded", 
                       "asset.variety.Unknown"
                       "asset.variety.Network")
    elif task == "asset.assets.variety":
    elif task == "action":
    elif task == "action.x.variety" 

    # Preprocessing
    X, y = preprocessing(df, veris_df,
                         PREDICTORS, 
                         TARGETS, 
                         INDUSTRY_ONE_HOT,
                         IMPUTER)
            

    ## Hyperparameter tuning for training separate classifiers for each 2nd level action
    if METRIC in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            grouped_search(X, y, 
                           metric=METRIC,
                           average=AVERAGING,
                           n_jobs_cv=N_JOBS_CV,
                           models=MODELS,
                           predictors=PREDICTORS, 
                           n_folds=N_FOLDS,
                           random_state=RANDOM_STATE,
                           readme=README,
                           experiment_id=experiment_id)