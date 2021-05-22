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
    
    veris_df = pd.read_csv(os.path.join(csv_dir, veris_df_csv),
                           index_col=0,
                           low_memory=False,
                           nrows=nrows)
    
    # df = pd.read_csv(os.path.join(csv_dir, collapsed),
    #                  low_memory=False,
    #                  index_col=0,
    #                  nrows=nrows)
    
    return veris_df

def create_veris_csv(json_dir=JSON_DIR,
                     csv_dir=CSV_DIR,
                     veris_df_csv=VERIS_DF):
    
    if  json_dir == None or csv_dir == None:
    
        logging.info("Need json and collapsed csv directories")
    
        exit()
    
    v = VERIS(json_dir=json_dir)
    
    veris_df = v.json_to_df(verbose=False)
    
    veris_df.to_csv(os.path.join(csv_dir, veris_df_csv))
    
    return veris_df

def mapper(x, asset_variety):
    
    splitted = x.split(" - ")
    
    splitted[0] = f"asset.assets.variety.{asset_variety}"
    
    return ".".join(splitted)

def check_y_statistics(y):
    
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

def ColToOneHot(collapsed, veris_df, father_col="action", replace=True) -> object:
    
    """ Transforms all features (whose names are children of the term defined
    from 'father_col' argument) of the collapsed version of the dataset to One Hot 
    Encodings using their values in the full veris_df version. A new dataset can be created
    for the encodings or alternatively the can be replaced on the original dataset. Args:
    
    Parameters
    ---------- 
    collapsed: DataFrame
        The collapsed version of the dataset that requires transformation of specific 
        columns. This version is normally produced by the json_to_collapsed R script
    
    veris_df: DataFrame
        The veris_df dataset

    father_col: str
        The prefix that defines which column will be transformed to One Hot Encodings
        e.g. father_col = 'action' gets transformed to action.Error, action.Hacking, ...

    replace: boolean 
        Choose weather to replace the column on the same dataframe or return a new one 
        with the encoded columns
    
    Returns
    -------
    Dataframe
        Initial dataframe with column replaced by encodings or fresh dataframe of encodings only      
    """

    if father_col.startswith("asset.assets.variety."):

        asset_variety = father_col.split(".")[-1]
        
        columns = veris_df.filter(like=f"asset.assets.variety.{asset_variety[0]} -").columns
        
        renamed_columns = columns.map(lambda x: mapper(x, asset_variety))
        
        renamer = dict(zip(columns.tolist(), renamed_columns.tolist()))
        
        veris_df.rename(mapper=renamer,
                        axis="columns",
                        inplace=True)
    if replace:
        
        collapsed_ = collapsed.copy()
        
        for attr in list(v.enum_summary(veris_df, father_col).iloc[:, 0]):
        
            sub = father_col + "." + attr
        
            collapsed_[sub] = veris_df[sub].astype(int)
        
        collapsed_.drop(columns=father_col, inplace=True)
        
        return collapsed_
    
    else:
    
        OneHot = pd.DataFrame()
    
        for attr in list(v.enum_summary(veris_df, father_col).iloc[:, 0]):
    
            sub = father_col + "." + attr
    
            OneHot[sub] = veris_df[sub]
    
        return OneHot.astype(int)

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

    logging.info('\nChecking whether to register model...\n')

    client = MlflowClient()

    current_run = mlflow.active_run().info.run_id

    current_model_data = client.get_run(current_run).data

    # lookup for previous version of model
    try:

        old_model_properties = dict(
            client.search_model_versions(
            f"name='{model_name}'"
            )[0])

    except IndexError:

        old_model_properties = None

    # if there is a previous model we need to compare and then update or not
    if old_model_properties != None:
        
        logging.info("\n Comparing with previous production model...\n ")
        
        old_model_data = client.get_run(
            old_model_properties['run_id']
            ).data
        
        if old_model_data.metrics['test_f1-macro'] < current_model_data.metrics['test_f1-macro'] \
            and current_model_data.metrics['test_hl'] >= -0.05 and current_model_data.metrics['test_hl'] <= 0:

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

    else: # in case there is no other model immediately register current model
        
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