import mlflow
import pandas as pd
import os, sys
from dotenv import load_dotenv
from verispy import VERIS
from sklearn.model_selection import train_test_split
import itertools
from mlflow.tracking import MlflowClient
import urllib.request
from zipfile import ZipFile
import logging

# Environment variables
load_dotenv() 

JSON_DIR = os.environ.get('JSON_DIR')

CSV_DIR = os.environ.get('CSV_DIR')

RCOLLAPSED = os.environ.get('R_COLLAPSED_CSV_NAME')

VERIS_DF = os.environ.get('BOOLEAN_CSV_NAME')

VERIS_DF_URL = os.environ.get('BOOLEAN_CSV_URL')

v = VERIS(json_dir=JSON_DIR)

def download_veris_csv(url=VERIS_DF_URL,
                       csv_dir=CSV_DIR,
                       veris_df_csv=VERIS_DF):
    """ 
    Downloads the official veris_df dataset (boolean VCDB dataset) 
    from the vz-risk github and returns it as DataFrame.

    Parameters
    ---------- 
        url:
            The remote path of the zipped folder containing the csv file
            (e.g. 'https://raw.githubusercontent.com/vz-risk/VCDB/master/data/csv/vcdb.csv.zip')
        
        csv_dir: str / path
            The local path to store the csv tono-serving

    Returns
    ---------- 
    DataFrame
        The veris_df dataset as Pandas DataFrame  

    """

    logging.info(f'Downloading from: {url}')
    
    urllib.request.urlretrieve(url, 
        os.path.join(csv_dir, 'vcdb.csv.zip'))

    with ZipFile(os.path.join(csv_dir, 'vcdb.csv.zip'), 'r') as zipObj:
        zipObj.extract(csv_dir)

    return pd.read_csv(os.path.join(csv_dir, veris_df_csv),
                       index_col=0,
                       low_memory=False)

def create_veris_csv(json_dir=JSON_DIR,
                     csv_dir=CSV_DIR,
                     veris_df_csv=VERIS_DF):
    """ 
    Creates veris_df type dataset (boolean VCDB dataset) and stores as csv.

    Parameters
    ---------- 
        json_dir: str / path
            The path of validated vcdb json files
        
        csv_dir: str / path
            The path to store the produced csv
        
        veris_df_csv: str
            The name of the csv file

    Returns
    ---------- 
    DataFrame
        The veris_df dataset as Pandas DataFrame  
    """

    if  json_dir == None or csv_dir == None:
    
        logging.info("Need json and collapsed csv directories")
    
        exit()
    
    v = VERIS(json_dir=json_dir)
    
    veris_df = v.json_to_df(verbose=False)
    
    veris_df.to_csv(os.path.join(csv_dir, veris_df_csv))
    
    return veris_df

def load_dataset(csv_dir=CSV_DIR, 
                 veris_df_csv=VERIS_DF, 
                 nrows=None):
    """ 
    Loads veris_df type csv (boolean VCDB dataset) from disk as a 
    Pandas DataFrame.

    Parameters
    ---------- 
        csv_dir: str / path
            The path to read the csv from
        
        veris_df_csv: str
            The name of the csv file

    Returns
    ---------- 
    DataFrame
        The loaded veris_df dataset as Pandas DataFrame  

    """    
    veris_df = pd.read_csv(os.path.join(csv_dir, veris_df_csv),
                           index_col=0,
                           low_memory=False,
                           nrows=nrows)
    
    return veris_df

def check_y_statistics(y):
    """ 
    Checks if output variable is statistically capable for training.

    Parameters
    ---------- 
        y: Pandas Series
            The output Series of the supervised task 
    Returns
    ---------- 
    Dict
        Dictionary of type of anomaly and statistics of y  

    """

    error_class = '-'
    
    if len(y) < 30:
    
        logging.info("Less than 30 samples")
    
        error_class = "few samples"
    
    elif sum(y) < len(y)/100:
    
        logging.info("Class imbalance> 1/20")
    
        error_class = "few positive instances"
    
    mlflow_tags = {'error_class': error_class,
                   'n_samples': len(y),
                   'n_positive': sum(y)}
    
    return mlflow_tags

def train_test_split_and_log(X, y, train_size, random_state):
    """ Performs train / test split and logs the datasets using the 
        MLflow tracking API 
    Parameters
    ---------- 
        X: DataFrame
            Input dataset
        
        y: Pandas Series or numpy array
            Output vector
        
        train_size: float
            Size of the training set as percentage of the whole dataset (0 < train_size <= 1)
        
        random_state: int
            Random state for the split 

    Returns
    ---------- 
    Case 1: Dataframe, Dataframe, Series, Series
        X_train, X_test, y_train, y_test
    
    Case 2 (train_size==1): Dataframe, Dataframe, str, str
        X_train, X_test, _, _

    """

    # Train / Test split
    if train_size < 1:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, 
                             y, 
                             train_size=train_size,
                             test_size=1-train_size,
                             shuffle=True,
                             stratify=y, 
                             random_state=random_state
                             )
    else:
        X_train = X
        
        y_train = y
        
        X_test = "all data was used as training set"
        
        y_test = "all data was used as training set"

    # Log datasets
    with open('X_train.csv', 'w', encoding='utf-8') as f:

        mlflow.log_artifact('X_train.csv')
        
        f.close()
    
    with open('y_train.csv', 'w', encoding='utf-8') as f:
    
        mlflow.log_artifact('y_train.csv')
    
        f.close()
    
    if isinstance(X_test, pd.DataFrame):
    
        with open('X_test.csv', 'w', encoding='utf-8') as f:
    
            mlflow.log_artifact('X_test.csv')
    
            f.close()  
    
        with open('y_test.csv', 'w', encoding='utf-8') as f:
    
            mlflow.log_artifact('y_test.csv')
    
            f.close() 

    return X_train, X_test, y_train, y_test


def mlflow_register_model(model_name):
    """ Compares the current model to the production model that is registered
        in mlflow models.
    Parameters
    ---------- 
        model_name: str
            The model name (same with target name)

    Returns
    ---------- 
    """
    logging.info('\nChecking whether to register model...\n')

    client = MlflowClient()

    current_run = mlflow.active_run().info.run_id

    current_model_data = client.get_run(current_run).data

    # lookup for previous version of model
    try:
        # last model version is the one in production
        old_model_properties = dict(
            client.search_model_versions(
            f"name='{model_name}'"
            )[-1])

    except IndexError:

        old_model_properties = None

    # if there is a previous model in production we need to compare and then update or not
    if old_model_properties != None and old_model_properties['current_stage'] == 'Production':
        
        logging.info("\n Comparing with previous production model...\n ")
        
        old_model_data = client.get_run(
            old_model_properties['run_id']
            ).data
        
        if old_model_data.metrics['test_f1-macro'] < current_model_data.metrics['test_f1-macro'] \
            and ((current_model_data.metrics['test_hl'] >= -0.05) or (current_model_data.metrics['test_hl'] >= old_model_data.metrics['test_hl'])) \
            and current_model_data.metrics['test_hl'] <= 0:

            # Register new model and transit it to production
            new_registered_model = mlflow.register_model(
                f"runs:/{current_run}/model", 
                model_name
                )

            client.transition_model_version_stage(
                name=model_name,
                version=new_registered_model.version,
                stage="Production"
                )

            # Archive old model
            client.transition_model_version_stage(
                name=model_name,
                version=str(int(new_registered_model.version) - 1),
                stage="Archived")
            
            logging.info('New model is accepted and put to production! \n')

        else:

            logging.info('New model is rejected. The previous one is kept in production! \n')

    else: # in case there is no other model in production immediately register current model
        
        logging.info('\n')
        mlflow.register_model(
            f"runs:/{current_run}/model", 
            model_name
            )
        logging.info('\n')
        
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Production"
            )
    return