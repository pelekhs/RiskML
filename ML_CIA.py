import sys
import os
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import ML_tuner_CV
import pandas as pd
from verispy import VERIS
import datetime

#from create_csv import create_veris_csv

JSON_DIR = '../VCDB/data/json/validated/'

CSV_DIR = "../csv/"

RESULTS_DIR = "./results/"

N_JOBS_CV = 6

PARAM_GRID = [dict(kernel=['rbf', 'linear'], C=[0.1, 1, 10, 100]),
              dict(bootstrap=[True], max_depth=[16, 20, 24, 30], max_features=['auto'], min_samples_leaf=[1, 2, 4],
                   min_samples_split=[12, 16, 20], n_estimators=[100, 400]),
              dict(),
              dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15], weights=['uniform', 'distance']),
              dict(max_iter=[100, 300]),
              dict()
             ]

def create_log_folder(pipeline, results_type="Tuning", results_root=RESULTS_DIR):
    now = datetime.datetime.now() + datetime.timedelta()
    pipeline_dir = os.path.join(results_root, pipeline))
    save_dir = os.path.join(pipeline_dir, results_type, str(now.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir)
    return save_dir

def ColToOneHot(collapsed, veris, father_col="action"):
    for attr in list(v.enum_summary(veris_df, father_col).iloc[:,0]):
        sub = father_col + "." + attr
        collapsed[sub] = veris[sub]
    collapsed.drop(columns=father_col, inplace=True)
    return collapsed

def OneHotbyTerm(collapsed, veris_df, term="Multiple"):
    for col, row in collapsed.items():
        if term in list(row):
            df = ColToOneHot(collapsed, veris_df, col)
    return df.reindex(sorted(df.columns), axis=1)


def knn_imputer(train):
    sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
    # start the KNN training
    return(fast_knn(train.values, k=30))

def subset(df, feat_columns, target_columns):
    features = df[feat_columns]
    output = df[target_columns]
    return features, output

# LOAD csv datasets

# create_veris_csv(JSON_DIR, os.path.join(CSV_DIR, "veris_df.csv"))
v = VERIS(json_dir=JSON_DIR)
veris_df = pd.read_csv(os.path.join(CSV_DIR, "veris_df.csv"),
                       index_col=0, low_memory=False)

# should normally start with the "Preprocessed.csv" produced by preprocessing.py
collapsed = pd.read_csv(os.path.join(CSV_DIR, "Rcollapsed.csv"),
                        sep=",", encoding='utf-8', index_col=0,
                        low_memory=False).reset_index(drop=True)

# filter out environmental incidents
collapsed = collapsed[collapsed["action"] != "Environmental"]

# NaN handling

values = {"action.malware.variety": "Unknown",
          "action.malware.vector": "Unknown",
          "action.malware.vector": "Unknown",
          "action.misuse.vector": "Unknown",
          "action.social.vector": "Unknown",
          "asset.variety": "Unknown",
          "asset.assets.variety": "Unknown",
          "pattern_collapsed": "skata"} # it is going to be replaced during OneHot
collapsed = collapsed.fillna(value=values)

# Features and target variables

features, targets = subset(collapsed,
                          ['action', 'action.error.variety', 'action.error.vector',
                           'action.hacking.variety', 'action.hacking.vector','action.malware.variety',
                           'action.malware.vector', 'action.misuse.variety', 'action.misuse.vector',
                           'action.physical.variety', 'action.physical.vector', 'action.social.target',
                           'action.social.variety', 'action.social.vector', 'asset.assets.variety',
                           'asset.variety', 'pattern_collapsed', 'timeline.incident.year',
                           'victim.industry.name', 'victim.orgsize'],
                          ['attribute.confidentiality', 'attribute.integrity', 'attribute.availability'])


# One Hot Encoding for Categorical Features

# Unfold to One Hot if column contains term "Multiple"
X = OneHotbyTerm(features.copy(), veris_df, "Multiple").reset_index(drop=False)
# victim.industry
lb = LabelBinarizer()
industry_one_hot_df = pd.DataFrame(data=lb.fit_transform(X["victim.industry.name"]),
                                   columns=["victim.industry." + name for name in lb.classes_])
X = pd.concat([X.drop(columns="victim.industry.name"), industry_one_hot_df], axis=1). \
    drop(columns="index")


# Label encode X

# orgsize
le = LabelEncoder()
le.fit(["Small", "Large", "Unknown"])
X["victim.orgsize"] = le.transform(X["victim.orgsize"])
# timeline.incident.year
X["timeline.incident.year"] = le.fit_transform(X["timeline.incident.year"])

# Numpyfy and label encode y
y_CIA = targets.astype(int)

# Feature Selection
#feature selection()

# Scaling & Train test split (same for all CIA)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train_CIA, y_test_CIA = \
    train_test_split(X, y_CIA.values, test_size=0.2,
                     train_size=0.8, shuffle=True)

# create folders for storing model info
now = datetime.datetime.now() + datetime.timedelta()
model_dir = os.path.join(RESULTS_DIR, str(now.strftime("%Y%m%d-%H%M%S")))
os.makedirs(model_dir)
os.makedirs(os.path.join(model_dir, "Tuning"))

# Hyperparameter Tuning
# Create log folder for hyperparameter tuning
save_dir = create_log_folder(pipeline="CIA", results_type="Tuning")

final_scores = {}
for (att, index) in zip(y_CIA.columns, range(len(y_CIA.columns))):
    print("\n************************\n{}\n************************".format(att))
    final_scores[att], _ = ML_tuner_CV(X_train, X_test,
                                       y_train_CIA[:, index], y_test_CIA[:, index],
                                       n_jobs_cv=N_JOBS_CV,
                                       param_grid=PARAM_GRID,
                                       save_CV_results_as_json=os.path.join(os.path.normpath(save_dir),
                                                                            'CV_results_' + att + '.json'))

# Evaluation of best ones