import sys
import os
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import ML_tuner_CV
import pandas as pd
from verispy import VERIS
from sklearn import tree
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

# Evaluation of best ones
y = y_CIA.loc[:, "attribute.confidentiality"]
clf = tree.DecisionTreeClassifier(max_depth = 3)
clf = clf.fit(X, y)

tree.plot_tree(clf.fit(X, y))

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.columns,
                                class_names=["Not C", "C"],
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("Confidentiality loss")

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=20, max_depth=2)
clf = clf.fit(X, y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
cols = X.columns[model.get_support()]
X_new = model.transform(X)
X_new = pd.DataFrame(data=X_new, columns=cols)

importance = list(clf.feature_importances_)
colum = list(X.columns)
out = pd.DataFrame(list(zip(colum, importance)), columns=['Feature', 'Importance'])

from sklearn.feature_selection import SelectKBest
kbest = SelectKBest(k=5)
X_new2 = kbest.fit_transform(X, y)
cols = X.columns[kbest.get_support()]
X_new2 = pd.DataFrame(data=X_new2, columns=cols)