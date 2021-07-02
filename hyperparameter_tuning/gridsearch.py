from sklearn.model_selection import GridSearchCV
import sys, os
import json
import pprint
import mlflow
import click
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import logging
from sklearn.pipeline import Pipeline
from distutils import util

# include parent directory in path
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)
import models

from evaluation import get_scorer
from utils import train_test_split_and_log, create_veris_csv
from etl import etl, etl_maker
from preprocessing import OneHotTransformer, myPCA


# Default values

# Parse default settings from training json
# with open('./gridsearch_config.json') as json_file:
#     default_arguments = json.load(json_file)
#     # str to bool
#     default_arguments['merge'] = util.strtobool(default_arguments['merge'])
#     default_arguments['models'] = models.models

def mlflow_gridsearch(X, y, 
                      estimator_dict,
                      predictors,
                      n_folds, 
                      metric,
                      averaging, 
                      n_jobs):
    """ 
    Given a dataset y|X, this function applies hyperparameter tuning on a given estimator
    on the VCDB datasets and loops over a dict of classifiers (see models.py). It then logs
    results to the MLflow server using the mlflow.sklearn autolog component.
       
    Parameters
    ---------- 
        X: Pandas DataFrame
            Input dataset

        y: Pandas Series
            Output vector (binary targets)
        
        estimator_dict: dict
            Dictionary that describes (see models.py) the particular classifier 
            that will be thyperparameter tuned.
        
        predictors: list
            List of column names to be used as predictors from the dataset 

        metric: str
            Metric to optimise. Choose from: "precision", "recall", "f1", 
            "hl", "accuracy", "auc"

        averaging: str
            Metric averaging method. Chose from: "micro", "macro", "weighted"
        
        n_folds: int
            Number of folds for cross validation during gridsearch
        
        n_jobs: int
            Number of cores to devote to GridsearchCV

    Returns
    ---------- 
    """

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
                models=models.models, 
                n_jobs_cv=1, 
                n_folds=5): 
    """ 
    Given a dataset y|X, this function loops over a dict of classifiers (see models.py).
    For each classifier it calls the mlflow_gridsearch function that performs
    hyperparameter search on the parameter grids also defined in models.py
       
    Parameters
    ---------- 
        X: Pandas DataFrame
            Input dataset

        y: Pandas Series
            Output vector (binary targets)
        
        predictors: list
            List of column names to be used as predictors from the dataset 

        metric: str
            Metric to optimise. Choose from: "precision", "recall", "f1", 
            "hl", "accuracy", "auc"

        averaging: str
            Metric averaging method. Chose from: "micro", "macro", "weighted"

        models: dict
            Dictionary of models to be tuned along with names and hyperparameter
            grids. (See models.py)
        
        n_folds: int
            Number of folds for cross validation during gridsearch
        
        n_jobs_cv: int
            Number of cores to devote to GridsearchCV

    Returns
    ---------- 
    """

    """  ML classifier hyperparameter tuning """
    
    # loop on estimators 
    for estimator_dict in models:
        # if estimator_dict["family_name"] == 'SVM':
        
        #     estimator_dict['parameter_grid']['predict_proba'] = [i for i in models.models
        #                 if i['family_name']==algo][0]['class'](**hyperparams)

        logging.info(f'\n{estimator_dict["family_name"]}\n--------------------')
        # mlflow run on estimators
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
                   'action.social.variety',
                   'default']),
#              default=default_arguments['task'],
              help="Learning task"
              )
@click.option("--imputer", "-i", 
              type=click.Choice(['dropnan', 'default']),
              multiple=False, 
#              default=default_arguments['imputer'],
              help="Imputation strategy"
              )
@click.option("--metric", "-m", 
              type=click.Choice(['auc', 'accuracy', 'precision', 'recall', 'f1', 'hl', 'default']),
#              default=default_arguments['metric'], 
              help="Metric to maximise during tuning."
              )
@click.option("--averaging", "-a",
              type=click.Choice(['micro', 'macro', 'weighted', 'default']),
#              default=default_arguments['averaging'],
              help="Method to compute aggregate metrics"
              )
@click.option("--n-folds", "-f", 
              type=int,
#              default=default_arguments['n_folds'],
              help="Number of folds for CV"
              )
@click.option("--n-jobs-cv", "-j", 
              type=int,
 #             default=default_arguments['n_jobs_cv'],
              help="Number of cores for GridsearchCV"
              )
@click.option("--random-state", "-r", 
              type=int,
#              default=default_arguments['random_state'],
              help="Random state for train/test splits"
              )
# @click.option("--train-size", "-t", 
#               type=float,
#               default=default_arguments['train_size'],
#               help="Training set percentage of the dataset size"
#               )
@click.option("--pca", "-p", 
              type=int,
#              default=default_arguments['pca'],
              help="Number of PCA components. 0 means no PCA"
              )
@click.option("--merge", "-m", 
              type=str,
#              default=default_arguments["merge"],
              help="Whether to merge Brute force, SQL injection and DoS columns for hacking \
                    and malware cases. Accepted values: ['y', 'yes', 't', 'true', 'on'] \
                    and ['n', 'no', 'f', false', 'off']"
              )

def run(task, metric, averaging, imputer, n_folds, n_jobs_cv, random_state, pca, merge):
    
    # Take care of boolean argument
    merge = util.strtobool(merge)

    # Load data
    veris_df = create_veris_csv()
    
    # Manage task predictors and features
    pp = etl_maker()

    # Manage task predictors and features
    _, targets = pp.get_task(task, veris_df)

    logging.info("Targets:", targets)
    
    metric_id = ("-".join([metric, averaging])) if (metric not in ['hl', 'accuracy']) else metric
    
    client = mlflow.tracking.MlflowClient()
    
    for target in targets:

        logging.info(f'\n************************\n{target}\n************************')
    
        with mlflow.start_run(run_name=f'{target} | {metric}-{averaging}') as run:
            # ETL
            target2 = target.split("- ")[-1] if "-" in target else target.split(".")[-1]
            X, y, predictors, target = etl(veris_df,
                                           task, 
                                           target2, 
                                           merge)
            logging.info(f'\nGridsearch over: {target}\n')

            # Log tags and params
            mlflow.set_tag("mlflow.runName", f'{target} | {metric}-{averaging}')

            mlflow.set_tags({'target': target, 
                             'predictors': predictors,
                             'n_samples': len(y),
                             'class_balance': sum(y)/len(y),
                             'tuning_metric': metric_id,
                             'imputer': imputer,
                             'merge': merge})

            # train / test if train percentage < 1
            X, _, y, _ = \
                train_test_split_and_log(X, 
                                         y, 
                                         train_size=1, 
                                         random_state=random_state)
            # scikit pipeline
            preprocessor = Pipeline(steps=[
                ('one_hot_encoder', OneHotTransformer(
                    feature_name='victim.industry.name', 
                    feature_labels=X['victim.industry.name'].unique().tolist())),
                ('PCA', myPCA(pca))])
        

            # Apply pipeline preprocessing transforms first
            X = preprocessor.fit_transform(X)

            # Single target tuning only on train data.
            ML_tuner_CV(X, 
                        y, 
                        predictors=predictors, 
                        metric=metric,
                        averaging=averaging,
                        n_jobs_cv=n_jobs_cv,
                        models=models.models,
                        n_folds=n_folds)
    return

if __name__ == '__main__':
    run()