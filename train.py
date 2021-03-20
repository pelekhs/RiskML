from mlflow import log_metric, log_param, log_artifacts
import click
import json
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score

import globals
from evaluation import evaluate
from preprocessing import preprocessing
from utils import get_task, load_datasets, check_y_statistics
import pprint


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

def train(X, y, estimator, hyperparams, train_size, n_folds, split_random_state):
    print("Training & evaluation...\n")
    # Train / Test split
    if train_size < 1:
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, 
                                train_size=train_size,
                                test_size=1-train_size,
                                shuffle=True,
                                stratify=y, 
                                random_state=split_random_state)
    else:
        X_train = X
        y_train = y

    # Log datasets
    with open('X_train.csv', 'w', encoding='utf-8') as f:
        X_train.to_csv(f)
        mlflow.log_artifact('X_train.csv')
        f.close()
    with open('y_train.csv', 'w', encoding='utf-8') as f:
        y_train.to_csv(f)
        mlflow.log_artifact('y_train.csv')
        f.close()
    if isinstance(X_test, pd.DataFrame):
        with open('X_test.csv', 'w', encoding='utf-8') as f:
            X_test.to_csv(f)
            mlflow.log_artifact('X_test.csv')
            f.close()  
        with open('y_test.csv', 'w', encoding='utf-8') as f:
            y_test.to_csv(f)
            mlflow.log_artifact('y_test.csv')
            f.close()  

    # # Fit estimator
    # estimator.fit(X_train, y_train)
    
    # # Predict target
    # y_pred = estimator.predict(X_test)
    # y_pred_proba = estimator.predict_proba(X_test)
    # print(y_pred_proba)

    # Log estimator params
    mlflow.log_params(estimator.get_params())

    # Evaluate for ALL METRICS and LOG!!!
    metrix_dict = evaluate(estimator, X, y)
    mlflow.log_metrics(metrix_dict)

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
@click.option("--target", "-tt", 
              type=str,
              default=default_arguments['target'],
              help="Specific target variable"
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
              help="training set size"
              )
@click.option("--split-random-state", "-rs",
              type=int, 
              default=default_arguments['split_random_state'],
              help="Random state for splitting train / test"
              )
@click.option("--n-folds", "-f", 
              type=int,
              default=default_arguments['n_folds'],
              help="Number of folds for CV if there training set is all dataset"
              )

def run(task, target, algo, hyperparams, imputer, train_size, split_random_state):

    # Process arguments
    hyperparams = json.loads(hyperparams)
    # activate probability for svm
    if algo == 'SVM':
        hyperparams['probability'] = True 
    estimator = [i for i in globals.MODELS 
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
    
    # Form target name
    target = " - ".join([task, target]) if task.startswith('asset.assets.variety') else \
            ".".join([task, target])

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
                         'imputer': imputer})

        # Train
        train(X, y, estimator, hyperparams, train_size, n_folds, split_random_state)

if __name__ == '__main__':
    run()