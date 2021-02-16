from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sys
import pandas as pd
import json
import pprint
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from preprocessing import create_log_folder
import os
from sklearn.metrics import recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from mlflow import log_metric, log_param, log_artifacts


MODELS = {
    'SVM': SVC(C=10, 
               kernel='rbf', 
               probability=True),
    'Knn': KNeighborsClassifier(n_neighbors=3, 
                                metric="hamming"),
    'RF': RandomForestClassifier(max_depth=16, 
                                 min_samples_leaf=1, 
                                 min_samples_split=16, 
                                 n_estimators=400),
    'LR': LogisticRegression(),
    'LGBM': LGBMClassifier(),
    'GNB': GaussianNB()
}

PARAM_GRID = [dict(kernel=['rbf', 'linear'],
                   C=[1, 10, 100]),
              dict(bootstrap=[True],
                   max_depth=[16, 20, 24],
                   max_features=['auto'],
                   min_samples_leaf=[1, 2, 4],
                   min_samples_split=[12, 16, 20],
                   n_estimators=[100, 400]),
              dict(),
              dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15],
                   weights=['uniform', 'distance'],
                   metric=["minkowski", "hamming"]),
              dict(max_iter=[100, 300]),
              dict()
             ]

# dict(hidden_layer_sizes=[(4, 4, 4), (4, 16, 16, 4), (16, 32, 32, 32, 16)],
#     activation=['relu', 'tanh'], solver=['sgd', 'adam'], batch_size=[32],
#     learning_rate=['adaptive']),dict(priors=[None, [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]),

def ML_tuner_CV(X_train, X_test, y_train, y_test, scorer="accuracy", average='macro',
                n_jobs_cv=-1, models=MODELS, param_grid=None, save_CV_results_as_json=None):
    """  ML classifier hyperparameter tuning """
    # Set scorers
    scorers = {
        'precision': make_scorer(precision_score, average=average),
        'recall': make_scorer(recall_score, average=average),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average=average),
        'auc': make_scorer(roc_auc_score, average=average)
    }
    # Set the parameters of each model by cross-validation gridsearch
    best_scores={}
    best_models={}
    best_params={}
    y_tests={}
    # Below I locate the best svm classifiers for each kernel and the best knn classifier and compute their test score
    print("Tuning hyper-parameters, based on accuracy, for various algorithms....")
    pp = pprint.PrettyPrinter(indent=2)
    skf = StratifiedKFold(n_splits=5)
    for (a, tp, name) in list(zip(models.values(), param_grid, models.keys())):
        print("\n{}\n--------------------".format(name))
        print("Parameter grid:\n {}".format(tp))
        clf = GridSearchCV(a, tp, cv=skf, 
                           scoring=scorers[scorer], 
                           refit=True,
                           n_jobs=n_jobs_cv)
        clf.fit(X_train, y_train)
        print("Mean CV performance of each parameter combination:")
        performance = pd.DataFrame(clf.cv_results_['params'])
        performance["Score"] = clf.cv_results_['mean_test_score']
        print(performance)
        print("\nBest parameters set found on training set:")
        pp.pprint(clf.best_params_)
        print("\nThe scores are computed on the full evaluation set:")
        #evaluate and store scores of estimators of each category on validation set
        score = clf.score(X_test, y_test)
        print(f"{scorer}-{average}: ", score)
        best_scores[name] = score
        best_models[name] = clf.best_estimator_
        best_params[name] = clf.best_params_
        y_tests[name] = y_test
    print("\n\n=====================================================")
    print("The best scores achieved by the algorithms are:\n{}\n".format(best_scores))
    if save_CV_results_as_json:
        results = {"Tuning metric": f"{scorer}-{average}", 
                   "Best_scores": best_scores,
                   "Best_params": best_params}
        with open(save_CV_results_as_json, 'w') as outfile:
            json.dump(results, outfile, indent=4)
    return best_scores, best_models, best_params

def grouped_tuning(X_train, X_test, y_trains, y_tests,  
                   models=MODELS, param_grid=PARAM_GRID, 
                   tune_metric="accuracy", average='macro',
                   results_dir="./results/", pipeline="Untitled", readme=True, 
                   n_jobs_cv=-1):
    ### Create log folder for hyperparameter tuning
    save_dir = create_log_folder(pipeline=pipeline,
                                 results_type="tuning",
                                 results_root=results_dir)
    # Logging
    if readme:
        results = {"Input features": X_train.columns.tolist(),
                   "Targets": y_trains.columns.tolist()}
        with open(os.path.join(os.path.dirname(save_dir), "ReadMe.txt"), 'w') as outfile:
            json.dump(results, outfile, indent=4)
    ### Tuning
    best_scores = {}
    best_models = {}
    best_params = {}
    # Instead of scikit multioutput classifier
    for target in y_trains.columns:
        print("\n************************\n{}\n************************".format(target))
        best_scores[target], best_models[target], best_params[target] = \
                ML_tuner_CV(X_train, X_test,
                            y_trains[target], y_tests[target],
                            scorer=tune_metric,
                            n_jobs_cv=n_jobs_cv,
                            param_grid=param_grid,
                            models=models,
                            average=average,
                            save_CV_results_as_json=os.path.join(os.path.normpath(save_dir),
                                                                                          target + '.json'))

    return best_scores, best_models, best_params


# if __name__=="__main__":
#
#     if len(sys.argv) < 5:
#         print("NEED AS INPUTS: X_train, X_test, y_train, y_test. Exiting...")
#         exit()
#
#     X_train = sys.argv[1]
#     X_test = sys.argv[2]
#     y_train = sys.argv[3]
#     y_test = sys.argv[4]
#
#     ML_tuner_CV(X_train, X_test, y_train, y_test)

