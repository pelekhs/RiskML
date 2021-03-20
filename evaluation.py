from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as classmetrics
from collections.abc import Iterable
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from sklearn.metrics import recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Hosmer Lemeshow
# See https://jbhender.github.io/Stats506/F18/GP/Group13.html
def hl_test(y_pred, y_true, g):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data
    Input: dataframe(data), integer(num of subgroups divided)
    Output: float
    '''
    y_pred_st = y_pred.sort_values('prob')
    y_pred_st['dcl'] = pd.qcut(y_pred_st['prob'], g)
    
    ys = y_pred_st['ViolentCrimesPerPop'].groupby(y_pred_st.dcl).sum()
    yt = y_pred_st['ViolentCrimesPerPop'].groupby(y_pred_st.dcl).count()
    yn = yt - ys
    
    yps = y_pred_st['prob'].groupby(y_pred_st.dcl).sum()
    ypt = y_pred_st['prob'].groupby(y_pred_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pval = 1 - chi2.cdf(hltest, g-2)
    
    return(pval)

def logit_p(skm, x):
    '''
    Print the p-value for sklearn logit model
   (The function written below is mainly based on the stackoverflow website -- P-value function for sklearn[3])
    
    Input: model, nparray(df of independent variables)
    
    Output: none
    '''
    pb = skm.predict_proba(x)
    n = len(pb)
    m = len(skm.coef_[0]) + 1
    coefs = np.concatenate([skm.intercept_, skm.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    result = np.zeros((m, m))
    for i in range(n):
        result = result + np.dot(np.trafrom sklearn.model_selection import StratifiedKFoldlt))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    pval = (1 - norm.cdf(abs(t))) * 2
    print(pd.DataFrame(pval, 
                       index=['intercept','population','pctUrban', 
                              'perCapInc', 'PctPopUnderPov'], 
                       columns=['p-value']))


def get_scorer(metric=None, average='macro'):
    scorers = {
        'precision': make_scorer(precision_score, average=average),
        'recall': make_scorer(recall_score, average=average),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average=average),
        'auc': make_scorer(roc_auc_score, average=average),
        'hl': make_scorer(hl_test, greater_is_better=False, needs_proba=True)
    }
    if metric in ['precision', 'recall', 'accuracy', 'f1', 'auc', 'hl']:
        return scorers[metric]
    elif metric=='all':
        return scorers
    else:
        print("Error: Please correctly define metric. Cannot get scorer!")

def evaluate(y_test, y_pred, y_pred_proba, n_folds):
    cv = StratifiedKFold(n_splits=n_folds)
    print("Evaluating...\n")
    for metric in ['precision', 'recall', 'f1', 'auc']:
        for average in ['macro', 'micro', 'weighted'];
            metrix_dict[metric+average] = \
                cross_val_score(estimator, X, y, 
                                get_scorer(metric=metric, average=average), 
                                cv=cv)
    metrix_dict["accuracy"] = cross_val_score(estimator, X, y, "accuracy", cv=cv)
    metrix_dict["hl_test"] = 0
    return metrix_dict


# def grouped_evaluation(X_train, X_test, y_trains, y_tests, models,
#                        evaluation_metric="f1", average='macro'):
#     print("\nEvaluation")
#     scores = {}
#     for target in y_trains.columns:
#         print("\n************************ {} ************************".format(target))
#         scores[target] = base_evaluator(X_train,
#                                         y_trains[target],
#                                         X_test,
#                                         y_tests[target],
#                                         metric=evaluation_metric,
#                                         average=average,
#                                         models=models)
#     return scores

# # Create scores dictionary for each algorithm
# def base_evaluator(X_train, y_train, X_test, y_test, models, metric="f1", average='macro'):
#     mdl=[]
#     results=[]
#     for model in models.keys():
#         print(f"Model: {model}")
#         est = models[model]
#         est.fit(X_train,  y_train)
#         mdl.append(model)
#         y_pred = est.predict(X_test)
#         # will also need probabilistic predictions to evaluate!
#         y_pred_prob = est.predict_proba(X_test)
#         results.append((est.score(X_test, y_test), y_pred))
#         print(metrics.classification_report(y_test, y_pred, digits=3))
#     results = [dict(zip(['Accuracy','y_pred'], i)) for i in results]

#     # At this point "scores" only contains accuracy and y_pred for each one of the best models chosen for each algorithm
#     scores = dict(zip(mdl, results))

#     # Enrich scores dictionary with the extra metrics
#     min_metric_class={}
#     max_metric_class={}
#     print("Evaluation Summary:")
#     for alg in scores.keys():
#         print ("\n", alg)
#         precision, recall, fscore, support = classmetrics(y_test, scores[alg]['y_pred'], average=average)
#         scores[alg]['precision'] = precision
#         scores[alg]['recall'] = recall
#         scores[alg]['f1'] = fscore
#         scores[alg]['support']  = support
#         print(metric)
#         print(alg)
#         print(f"{metric}-{average}  : {scores[alg][metric]}")
#         min_metric_class[alg] = np.argmin(scores[alg][metric])
#         max_metric_class[alg] = np.argmax(scores[alg][metric])
#     if isinstance(scores[alg][metric], Iterable) and len(scores[alg][metric]) > 1: # case of non-averaged metrics
#         print(f"\nWorst performance class for each classifier based on {metric}:")
#         print(min_metric_class)
#         print(f"Best performance class for each classifier based on {metric}:")
#         print(max_metric_class)
#     return scores