from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as classmetrics
from collections.abc import Iterable

# Create scores dictionary for each algorithm
def base_evaluator(X_train, y_train, X_test, y_test, models, metric="recall", average=None):
    mdl=[]
    results=[]
    for model in models.keys():
        print(f"Model: {model}")
        est = models[model]
        est.fit(X_train,  y_train)
        mdl.append(model)
        y_pred = est.predict(X_test)
        results.append((est.score(X_test, y_test), y_pred))
        print(metrics.classification_report(y_test, y_pred, digits=3))
    results = [dict(zip(['Accuracy','y_pred'], i)) for i in results]

    # At this point "scores" only contains accuracy and y_pred for each one of the best models chosen for each algorithm
    scores = dict(zip(mdl, results))

    # Enrich scores dictionary with the extra metrics
    min_metric_class={}
    max_metric_class={}
    print("Evaluation Summary:")
    for alg in scores.keys():
        print ("\n", alg)
        precision, recall, fscore, support = classmetrics(y_test, scores[alg]['y_pred'], average=average)
        scores[alg]['precision'] = precision
        scores[alg]['recall'] = recall
        scores[alg]['f1'] = fscore
        scores[alg]['support']  = support
        print(f"{metric}-{average}  : {scores[alg][metric]}")
        min_metric_class[alg] = np.argmin(scores[alg][metric])
        max_metric_class[alg] = np.argmax(scores[alg][metric])
    if isinstance(scores[alg][metric], Iterable) and len(scores[alg][metric]) > 1: # case of non-averaged metrics
        print(f"\nWorst performance class for each classifier based on {metric}:")
        print(min_metric_class)
        print(f"Best performance class for each classifier based on {metric}:")
        print(max_metric_class)
    return scores

def grouped_evaluation(X_train, X_test, y_trains, y_tests, models,
                       evaluation_metric="accuracy", average=None):
    print("\nEvaluation")
    scores = {}
    for target in y_trains.columns:
        print("\n************************ {} ************************".format(target))
        scores[target] = base_evaluator(X_train,
                                        y_trains[target],
                                        X_test,
                                        y_tests[target],
                                        metric=evaluation_metric,
                                        average=average,
                                        models=models)
    return scores