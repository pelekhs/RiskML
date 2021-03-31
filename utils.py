import mlflow
import pandas as pd
import os, sys
from dotenv import load_dotenv
from verispy import VERIS
from sklearn.model_selection import train_test_split
import itertools

# Environment variables
load_dotenv() 
JSON_DIR = os.environ.get('JSON_DIR')
CSV_DIR = os.environ.get('CSV_DIR')
RCOLLAPSED = os.environ.get('R_COLLAPSED_CSV_NAME')
VERIS_DF = os.environ.get('BOOLEAN_CSV_NAME')

v = VERIS(json_dir=JSON_DIR)

def load_datasets(csv_dir=CSV_DIR, veris_df_csv=VERIS_DF, rcollapsed=RCOLLAPSED):
    veris_df = pd.read_csv(os.path.join(csv_dir, veris_df_csv),
                           index_col=0,
                           low_memory=False)

    df = pd.read_csv(os.path.join(csv_dir, rcollapsed),
                     low_memory=False,
                     index_col=0)
    return df, veris_df

def create_veris_csv(json_dir=JSON_DIR,
                     csv_dir=CSV_DIR,
                     veris_df_csv=VERIS_DF):
    if  json_dir == None or csv_dir == None:
        print("Need json and collapsed csv directories")
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
    if len(y) < 50:
        print("Few samples")
        error_class = "few samples"
    elif sum(y) < len(y)/10:
        print("Few positive instances")
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

def get_task(task, veris_df):
    
    """ Defines the names of the predictor and target columns of veris_df dataset
        according to a selected task. Args:
    
    Parameters
    ---------- 
        task: str
            Opted task for learning
        
        veris_df: DataFrame 
            The veris_df dataset
    
    Returns
    ---------- 
    list
        a list of strings that contain the predictor columns
    list
        a list of strings that contain the target columns
    """

    if task == 'attribute':
        c = veris_df.filter(items=['attribute.Confidentiality']).columns
        i = veris_df.filter(items=['attribute.Integrity']).columns
        a = veris_df.filter(items=['attribute.Availability']).columns
        cia = list(itertools.chain(c, i, a))
        targets = cia
        predictors = ['action', 
                      'action.x.variety', 
                      'victim.industry', 
                      'victim.orgsize',
                      'asset.variety',
                      'asset.assets.variety'
                      ]
        return predictors, targets

    targets = veris_df.filter(like=task).columns.tolist()
    targets = [x for x in targets if \
            "Other" not in x and 'Unknown' not in x]
    if task == "asset.variety":
        predictors = ['action', 
                        'action.x.variety', 
                        'victim.industry', 
                        'victim.orgsize']
        targets.remove("asset.variety.Embedded")
        targets.remove("asset.variety.Network")
    elif task.startswith("asset.assets.variety."):
        predictors = ['action', 
                        'action.x.variety', 
                        'victim.industry', 
                        'victim.orgsize',
                        'asset.variety']
    elif task == 'action':
        predictors = ['asset.variety', 
                    'asset.assets.variety', 
                    'victim.industry', 
                    'victim.orgsize']
        targets = ["action.Hacking", 
                "action.Malware", 
                "action.Error", 
                "action.Misuse",
                "action.Physical", 
                "action.Social"]
    elif task.startswith("action") and task.endswith("variety"):
        predictors = ['asset.variety', 
                    'asset.assets.variety', 
                    'action',  
                    'victim.industry', 
                    'victim.orgsize',
                    'asset.variety']
    return predictors, targets

# def yield_artifacts(run_id, path=None):
#     """Yield all artifacts in the specified run"""
#     client = mlflow.tracking.MlflowClient()
#     for item in client.list_artifacts(run_id, path):
#         if item.is_dir:
#             yield from yield_artifacts(run_id, item.path)
#         else:
#             yield item.path


# def fetch_logged_data(run_id):
#     """Fetch params, metrics, tags, and artifacts in the specified run"""
#     client = mlflow.tracking.MlflowClient()
#     data = client.get_run(run_id).data
#     # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
#     tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = list(yield_artifacts(run_id))
#     return {
#         "params": data.params,
#         "metrics": data.metrics,
#         "tags": tags,
#         "artifacts": artifacts,
#     }

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
                train_test_split(X, y, 
                                train_size=train_size,
                                test_size=1-train_size,
                                shuffle=True,
                                stratify=y, 
                                random_state=random_state)
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