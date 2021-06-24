
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB

models = [
    {'class': SVC,
     'family_name': 'SVM',
     'parameter_grid': {'C': [1, 10, 100], 'kernel': ['rbf', 'linear'], 'predict_proba':[True, False]}},
    {'class': KNeighborsClassifier,
     'family_name': 'KNN',
     'parameter_grid': {'metric': ['minkowski', 'hamming'],
                        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
                        'weights': ['uniform', 'distance'],
                        'predict_proba':[True, False]}},
    {'class': RandomForestClassifier,
     'family_name': 'RF',
     'parameter_grid': {'bootstrap': [True],
                        'max_depth': [12, 16, 20, 24, 28],
                        'max_features': ['auto'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [12, 16, 20],
                        'n_estimators': [100, 400],
                        'predict_proba':[True, False]}},
    {'class': LogisticRegression,
     'family_name': 'LR',
     'parameter_grid': {'max_iter': [100, 300]}},
    {'class': LGBMClassifier, 'family_name': 'LGBM', 'parameter_grid': {}},
    {'class': GaussianNB, 'family_name': 'GNB', 'parameter_grid': {}}
        ]