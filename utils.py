import mlflow
import pandas as pd
import os, sys
from globals import CSV_DIR, JSON_DIR
from verispy import VERIS
from sklearn.model_selection import train_test_split

v = VERIS(json_dir=JSON_DIR)

def load_datasets():
    veris_df = pd.read_csv(os.path.join(CSV_DIR, "veris_df.csv"),
                           index_col=0,
                           low_memory=False)

    df = pd.read_csv(os.path.join(CSV_DIR, "etl_product.csv"),
                     low_memory=False,
                     index_col=0)
    return df, veris_df

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

def get_task_old(task, veris_df, level1=None):
    if task == "asset.variety":
        predictors = ['action', 
                      'action.x.variety', 
                      'victim.industry', 
                      'victim.orgsize']
        targets = veris_df.filter(like="asset.variety.").columns.tolist()
        targets.remove("asset.variety.Embedded")
        targets.remove("asset.variety.Unknown")
        targets.remove("asset.variety.Network")
    elif task.startswith("asset.assets.variety."):
        predictors = ['action', 
                      'action.x.variety', 
                      'victim.industry', 
                      'victim.orgsize',
                      'asset.variety']
        asset_variety = task.split(".")[-1]
        targets = veris_df.filter(like=f"asset.assets.variety.{asset_variety[0]} -") \
                          .columns.tolist()
        targets = list(map(lambda x: mapper(x, asset_variety), targets))
        targets = [x for x in targets if \
                    "Other" not in x and 'Unknown' not in x ]
        print(targets)
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
        targets = veris_df.filter(items=task) \
                            .columns.tolist()
        targets = [x for x in targets if \
                    "Other" not in x and 'Unknown' not in x ]
    return predictors, targets

def get_task(task, veris_df, level1=None):
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

def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }

def train_test_split_and_log(X, y, train_size, random_state):
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
    return X_train, X_test, y_train, y_test