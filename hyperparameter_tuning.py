
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import sys
import pandas as pd
import json
import pprint
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from preprocessing import create_log_folder
import os
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix

# dict(hidden_layer_sizes=[(4, 4, 4), (4, 16, 16, 4), (16, 32, 32, 32, 16)],
#     activation=['relu', 'tanh'], solver=['sgd', 'adam'], batch_size=[32],
#     learning_rate=['adaptive']),dict(priors=[None, [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]),

MODELS = {'SVM': SVC(),
          'Random Forest': RandomForestClassifier(n_jobs=-1),
          'LightGBM': LGBMClassifier(n_jobs=-1),
          'kNN': KNeighborsClassifier(n_jobs=-1),
          'Logistic Regression': LogisticRegression(n_jobs=-1),
          'Gaussian Naive Bayes': GaussianNB(),
          }

def ML_tuner_CV(X_train, X_test, y_train, y_test, scorer="accuracy", average=None,
                n_jobs_cv=-1, param_grid=None, save_CV_results_as_json=None):
    """  ML classifier hyperparameter tuning """
    # Set scorers
    scorers = {
        'precision': make_scorer(precision_score, average=average),
        'recall': make_scorer(recall_score, average=average),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average=average)
    }
    # Set the parameters of each model by cross-validation gridsearch
    best_scores=[]
    params={}
    y_tests={}
    # Below I locate the best svm classifiers for each kernel and the best knn classifier and compute their test score
    print("Tuning hyper-parameters, based on accuracy, for various algorithms....")
    pp = pprint.PrettyPrinter(indent=2)
    skf = StratifiedKFold(n_splits=5)
    for (a, tp, name) in list(zip(MODELS.values(), param_grid, MODELS.keys())):
        print("\n{}\n--------------------".format(name))
        print("Parameter grid:\n {}".format(tp))
        clf = GridSearchCV(a, tp, cv=skf, scoring=scorers[scorer], n_jobs=n_jobs_cv)
        clf.fit(X_train, y_train)
        print("Mean CV performance of each parameter combination:")
        performance = pd.DataFrame(clf.cv_results_['params'])
        performance["Score"] = clf.cv_results_['mean_test_score']
        print(performance)
        print("\nBest parameters set found on training set:")
        pp.pprint(clf.best_params_)
        params[name] = clf.best_params_
        print("\nThe scores are computed on the full evaluation set:")
        #evaluate and store scores of estimators of each category on validation set
        score = clf.score(X_test, y_test)
        print(f"{scorer}-{average}: ", score)
        best_scores.append(score)
        y_tests[name] = y_test
    final_scores = dict(zip(list(MODELS.keys()), best_scores))
    print("\n\n=====================================================")
    print("The best accuracies achieved by the algorithms are:\n{}\n".format(final_scores))
    if save_CV_results_as_json:
        results = {"Tuning metric": f"{scorer}-{average}",
                   "Best_scores": final_scores,
                   "Best_params": params}
        with open(save_CV_results_as_json, 'w') as outfile:
            json.dump(results, outfile, indent=4)
    return final_scores, params

def grouped_tuning(X_train, X_test, y_trains, y_tests, results_dir, pipeline,
                param_grid, n_jobs_cv, tune_metric="accuracy", average=None, readme=None):
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
    final_scores = {}
    params = {}
    # Instead of scikit multioutput classifier
    for target in y_trains.columns:
        print("\n************************\n{}\n************************".format(target))
        final_scores[target], params[target] = ML_tuner_CV(X_train, X_test,
                                                     y_trains[target], y_tests[target],
                                                     scorer=tune_metric,
                                                     n_jobs_cv=n_jobs_cv,
                                                     param_grid=param_grid,
                                                     average=average,
                                                     save_CV_results_as_json=os.path.join(os.path.normpath(save_dir),
                                                                                          target + '.json'))

    return final_scores, params


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

