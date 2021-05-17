import click
import json
import mlflow
import logging
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score
from lightgbm import LGBMClassifier
import pprint
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

import models
from evaluation import evaluate_cv, evaluate
from etl import etl
from preprocessing import OneHotTransformer, myPCA
from utils import create_veris_csv, load_dataset, download_veris_csv, check_y_statistics, train_test_split_and_log, mlflow_register_model
from inference import  ModelOut, mlflow_serve_conda_env

# Reset mlflow tracking uri
load_dotenv()

# env variables
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')


# Default values
default_arguments = {
    'task': 'asset.variety',
    'target': ' ',
    'algo': 'LGBM',
    'imputer': 'dropnan',
    'merge': 'True',
    'split_random_state': None,
    'train_size': 0.8,
    'n_folds': 0,
    'pca': 0
}

def train_evaluate(X, y, estimator, hyperparams, train_size, 
                   n_folds, split_random_state, n_components):
    
    logging.info("Checking capability of training")
    
    error_tags = check_y_statistics(y)

    mlflow.set_tags(error_tags)
    
    # skip loop if y is small
    error = error_tags['error_class']
    
    if error != '-':
        
        raise(ValueError(f"{error}, preferrably seek another way of learning!\n"))    
    
    else:
    
        logging.info("Passed!\n")

    if train_size < 1:
        
        logging.info("Splitting dataset\n")
        
        X_train, X_test, y_train, y_test = \
            train_test_split_and_log(X, y, train_size, split_random_state)

    else:

        X_train = X

        y_train = y

    logging.info("Preprocessing, training & evaluation\n")

    pipeline = Pipeline(steps=[
                    ('one_hot_encoder', OneHotTransformer(
                        feature_name='victim.industry.name', 
                        feature_labels=X['victim.industry.name'].unique().tolist())),
                    ('PCA', myPCA(n_components)), 
                    ('classifier', estimator)
                    ])

    if np.allclose(train_size, 1):  # CV evaluation for train_size = 1
        
        metrix_dict = evaluate_cv(pipeline, X_train, y_train, 
                                  n_folds, split_random_state)
        # Refit model to log it afterwards
        model = pipeline.fit(X_train, y_train) 
    
    else: # CV evaluation for train_size < 1
            
        model = pipeline.fit(X_train, y_train)
    
        y_pred_proba = model.predict_proba(X_test)
        
        y_pred = model.predict(X_test)
        
        metrix_dict = evaluate(y_test, y_pred, y_pred_proba)

    # Get model signature and log model
    signature = infer_signature(X, model.predict_proba(X.iloc[0:2]))

    mlflow.pyfunc.log_model(artifact_path="model",
                            python_model=ModelOut(model=pipeline),
                            code_path=['inference.py'],
                            conda_env=mlflow_serve_conda_env, 
                            input_example=X.head(1))

    # Log parameters of used estimator
    mlflow.log_params(estimator.get_params())
    
    # Log metrix
    mlflow.log_metrics(metrix_dict)

    return metrix_dict

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
@click.option("--merge", "-m", 
              type=bool,
              default=default_arguments["merge"],
              help="Whether to merge Brute force, SQL injection and DoS columns for hacking and malware cases"
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
@click.option("--pca", "-p", 
              type=int,
              default=default_arguments['pca'],
              help="Number of PCA components. 0 means no PCA"
              )

def train_evaluate_register(task, 
                            target, 
                            algo, 
                            hyperparams, 
                            imputer, 
                            merge,
                            train_size, 
                            split_random_state, 
                            n_folds,
                            pca):

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logging.info("Current tracking uri: {}".format(mlflow.get_tracking_uri()))

    # Process arguments
    hyperparams = json.loads(hyperparams)
    
    if np.allclose(train_size, 1) and n_folds < 2:
        mlflow.set_tags({'error class': 'train<1 & n_folds<2'})
        
        raise(ValueError('\nSelect train_size < 1 or n_folds >= 2.\n'))

    # Activate probability for svm
    if algo == 'SVM':
        
        hyperparams['probability'] = True 
    
    estimator = [i for i in models.models
                 if i['family_name']==algo][0]['class'](**hyperparams)
    
    # Load data
    #veris_df = download_veris_csv()
    #veris_df = load_dataset()
    veris_df = create_veris_csv()

    # Start training workflow and logs
    with mlflow.start_run(run_name=target):
        
        # ETL
        X, y, predictors, target = etl(veris_df,
                                       task, 
                                       target, 
                                       merge)

        logging.info(f'\nTraining {algo} on: {target}\n')
        
        # Tags
        mlflow.set_tag("mlflow.runName", f'{target}')
    
        mlflow.set_tags({'target': target, 
                         'predictors': predictors, 
                         'n_samples': len(y),
                         'class_balance': sum(y)/len(y),
                         'imputer': imputer,
                         'merge': merge
                         }
                       )

        # Train
        train_evaluate(X, 
                       y, 
                       estimator, 
                       hyperparams, 
                       train_size, 
                       n_folds, 
                       split_random_state, 
                       pca)

        # Update Model Registry if model is better than current
        mlflow_register_model(model_name=target)

    return

if __name__ == '__main__':

    train_evaluate_register()
    