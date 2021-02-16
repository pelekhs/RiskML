global JSON_DIR
global CSV_DIR
global RESULTS_DIR
JSON_DIR = '../VCDB/data/json/validated/'
CSV_DIR = "../csv/"
RESULTS_DIR = "./results/"

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB

global MODELS
MODELS = [
          {
           'family_name': 'SVM', 
           'class': SVC(),
           'parameter_grid': {
                              'kernel': ['rbf', 'linear'], 
                              'C': [1, 10, 100]
                             }
          },
          {
           'family_name': 'KNN',
           'class': KNeighborsClassifier(),
           'parameter_grid': {
                              'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
                              'weights': ['uniform', 'distance'],
                              'metric': ["minkowski", "hamming"]
                             }
          },
          {
           'family_name': 'RF', 
           'class': RandomForestClassifier(),
           'parameter_grid': {
                              'bootstrap': [True],
                              'max_depth': [12, 16, 20, 24, 28],
                              'max_features': ['auto'],
                              'min_samples_leaf': [1, 2, 4],
                              'min_samples_split': [12, 16, 20],
                              'n_estimators':[100, 400]
                             }
          },
          {
           'family_name': 'LR', 
           'class': LogisticRegression(),
           'parameter_grid': {
                              'max_iter': [100, 300]
                             }
          },
          {
           'family_name': 'LGBM', 
            'class': LGBMClassifier(),
            'parameter_grid': {}
          },
          {
           'family_name': 'GNB', 
           'class': GaussianNB(),
           'parameter_grid': {}
          }
         ]

# global PARAM_CUBE
# PARAM_CUBE = [
#               dict(kernel=['rbf', 'linear'],
#                    C=[1, 10, 100]),
#               dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15],
#                    weights=['uniform', 'distance'],
#                    metric=["minkowski", "hamming"]),
#               dict(bootstrap=[True],
#                    max_depth=[12, 16, 20, 24, 28],
#                    max_features=['auto'],
#                    min_samples_leaf=[1, 2, 4],
#                    min_samples_split=[12, 16, 20],
#                    n_estimators=[100, 400]),
#               dict(max_iter=[100, 300]),
#               dict(),
#               dict()
#              ]