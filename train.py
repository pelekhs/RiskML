import click
import json
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score
import pprint
import os

import models
from evaluation import evaluate_cv
from preprocessing import preprocessing
from utils import get_task, load_datasets, check_y_statistics, train_test_split_and_log


# Default values
default_arguments = {
    'task': 'asset.variety',
    'target': ' ',
    'algo': 'LR',
    'imputer': 'dropnan',
    'split_random_state': 0,
    'train_size': 1,
    'industry_one_hot': True,
    'n_folds': 5
}

def train_evaluate(X, y, estimator, hyperparams, train_size=1, n_folds=5, split_random_state=None):
    
    print("Splitting dataset...\n")
    X_train, X_test, y_train, y_test = \
        train_test_split_and_log(X, y, train_size, split_random_state)
    
    print("Training & evaluation...\n")

    # evaluation for train_size = 1
    if train_size == 1:
        metrix_dict = evaluate_cv(estimator, X, y, n_folds, split_random_state)
        mlflow.log_metrics(metrix_dict)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(metrix_dict)

        # Refit model to whole dataset and log
        model = estimator.fit(X, y)
        signature = infer_signature(X, model.predict_proba(X.iloc[0:2]))
        mlflow.sklearn.log_model(sk_model=model, 
                                 artifact_path="model",
                                 signature=signature)
    else:
        print ("Not ready to handle test set! Exitting...")
        exit()
    # /* To be implemented for simple train/test scoring (train_size < 1)*/
    # else:
        # estimator.fit(X_train, y_train)
        # y_pred = estimator.predict(X_test)
        # y_pred_proba = estimator.predict_proba(X_test)
        # metrix_dict = evaluate(y_test, y_pred, y_pred_prob)
    
    # Log estimator params
    mlflow.log_params(estimator.get_params())
    return

@click.command()
@click.option("--task", "-t", 
              type=click.Choice(
                  ['attribute',
                   'asset.variety',
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
@click.option("--target", "-tt", 
              type=str,
              default=default_arguments['target'],
              help="Specific target variable. Omit this option to get list of options according to task"
              )
@click.option("--algo", "-a", 
              type=click.Choice(
                  ['SVM', 
                  'RF',
                  'LR', 
                  'GNB',
                  'LGBM', 
                  'KNN']),
              multiple=False,
              default=default_arguments['algo'],
              help="Algorithm"
              )
@click.option("--hyperparams", "-h", 
              type=str,
              default='{}',
              help=""" "Hyperapameters of algorithm. e.g. '{"C": 1, "gamma": 0.1}' """
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan']),
              default=default_arguments["imputer"],
              help="Independent targets of the model"
              )
@click.option("--train-size", "-ts", 
              type=float, 
              default=default_arguments['train_size'],
              help="Training set size"
              )
@click.option("--split-random-state", "-rs",
              type=int, 
              default=default_arguments['split_random_state'],
              help="Random state for splitting train / test or cv"
              )
@click.option("--n-folds", "-f", 
              type=int,
              default=default_arguments['n_folds'],
              help="Number of folds for CV if there training set is all dataset"
              )

def run(task, target, algo, hyperparams, imputer, train_size, split_random_state, n_folds):

    # Process arguments
    hyperparams = json.loads(hyperparams)
    
    # activate probability for svm
    if algo == 'SVM':
        hyperparams['probability'] = True 
    estimator = [i for i in models.models
                 if i['family_name']==algo][0]['class'](**hyperparams)
    
    # Load data
    df, veris_df = load_datasets()
    
    # Manage task predictors and features
    predictors, targets = get_task(task, veris_df)

    # Let user choose specific target if not already done it from cli
    if target == ' ':
        pp = pprint.PrettyPrinter(indent=4)
        choices = dict(zip(range(len(targets)), targets))
        printer = pp.pprint(choices)
        target = choices[int(input(f"Select a target from the list above...\n{printer}\n"))]
    else:
    # Form target name
        target = " - ".join([task, target]) if task.startswith('asset.assets.variety') \
                                            else ".".join([task, target])

    # Start training workflow and logs
    print(f'\nTraining for: {target}...\n')

    with mlflow.start_run(run_name=target):

        # Preprocessing
        X, y = preprocessing(df, veris_df,
                             predictors, 
                             target, 
                             default_arguments['industry_one_hot'],
                             imputer)

        # Tags
        mlflow.set_tag("mlflow.runName", f'{target}')
        mlflow.set_tags({'target': target, 
                         'predictors': predictors, 
                         'n_samples': len(y),
                         'class_balance': sum(y)/len(y),
                         'imputer': imputer})

        # Train
        train_evaluate(X, y, estimator, hyperparams, train_size, n_folds, split_random_state)

if __name__ == '__main__':
    run()