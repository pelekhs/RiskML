from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import sys
import pandas as pd

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

PARAM_GRID = [dict(kernel=['rbf'], C=[1, 10, 100]),
              dict(bootstrap=[True], max_depth=[8, 12, 16], max_features=['auto', 5, 10], min_samples_leaf=[2, 4, 8],
                   min_samples_split=[4, 12, 16], n_estimators=[100, 300]),
              dict(),
              dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15], weights=['uniform', 'distance']),
              dict(max_iter=[200]),
              dict()]


def ML_tuner_CV(X_train, X_test, y_train, y_test, n_jobs_cv):
    """  ML classifier hyperparameter tuning """
    #Set the parameters of each model by cross-validation gridsearch
    best_scores=[]
    params=[]
    y_tests={}
    #Below I locate the best svm classifiers for each kernel and the best knn classifier and compute their score
    #on the test set
    print("Tuning hyper-parameters, based on accuracy, for various algorithms....")
    for (a, tp, name) in list(zip(MODELS.values(), PARAM_GRID, MODELS.keys())):
        print("\n{}\n------".format(name))
        print("Parameter grid:\n {}".format(tp))
        clf = GridSearchCV(a, tp, cv = 5, scoring = 'accuracy', n_jobs=n_jobs_cv)
        clf.fit(X_train, y_train)
        print("Mean CV performance of each parameter combination:")
        performance = pd.DataFrame(clf.cv_results_['params'])
        performance["Score"] = clf.cv_results_['mean_test_score']
        print(performance)
        print("\nBest parameters set found on training set:")
        print(clf.best_params_)
        params.append(clf.best_params_)
        print("\nThe scores are computed on the full evaluation set:")
        #evaluate and store scores of estimators of each category on validation set
        score = clf.score(X_test, y_test)
        print("Accuracy:", score)
        best_scores.append(score)
        y_tests[name] = y_test
    final_scores = dict(zip(list(MODELS.keys()), best_scores))
    print("\n\n================================================")
    print("The best accuracies achieved by the algorithms are:\n{}\n".format(final_scores))
    return final_scores, y_tests


if __name__=="__main__":

    if len(sys.argv) < 5:
        print("NEED AS INPUTS: X_train, X_test, y_train, y_test. Exiting...")
        exit()

    X_train = sys.argv[1]
    X_test = sys.argv[2]
    y_train = sys.argv[3]
    y_test = sys.argv[4]

    ML_tuner_CV(X_train, X_test, y_train, y_test)
