import sys
import os
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import ML_tuner_CV
import pandas as pd
from verispy import VERIS
import graphviz
#from create_csv import create_veris_csv

JSON_DIR = '../VCDB/data/json/validated/'

CSV_DIR = "../csv/"

N_JOBS_CV = 6

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

collapsed = pd.read_csv(os.path.join(CSV_DIR, "Rcollapsed.csv"),
                        sep=",", encoding='utf-8', index_col=0,
                        low_memory=False).reset_index(drop=True)

# filter out environmental incidents
collapsed = collapsed[collapsed["action"] != "Environmental"]

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
# NaN handling

values = {"action.malware.variety": "Unknown",
          "action.malware.vector": "Unknown",
          "action.malware.vector": "Unknown",
          "action.misuse.vector": "Unknown",
          "action.social.vector": "Unknown",
          "asset.variety": "Unknown",
          "asset.assets.variety": "Unknown",
          "pattern_collapsed": "skata"} # it is going to be replaced during OneHot
features = features.fillna(value=values)

# One Hot Encoding for Categorical Features

# Unfold to One Hot if column contains term "Multiple"
X = OneHotbyTerm(features.copy(), veris_df, "Multiple").reset_index(drop=False)
# victim.industry
lb = LabelBinarizer()
industry_one_hot_df = pd.DataFrame(data=lb.fit_transform(X["victim.industry.name"]), columns=lb.classes_)
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

# Tuning
final_scores = {}
for (att, index) in zip(y_CIA.columns, range(len(y_CIA.columns))):
    print("\n************************\n{}\n************************".format(att))
    final_scores[att], _ = ML_tuner_CV(X_train, X_test,
                                         y_train_CIA[:, index], y_test_CIA[:, index],
                                         n_jobs_cv=N_JOBS_CV)


# Evaluation of best ones