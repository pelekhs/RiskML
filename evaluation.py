from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as classmetrics
from collections.abc import Iterable
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer

# Hosmer Lemeshow
# See https://jbhender.github.io/Stats506/F18/GP/Group13.html
def hl_test(data, g):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data
    Input: dataframe(data), integer(num of subgroups divided)
    Output: float
    '''
    data_st = data.sort_values('prob')
    data_st['dcl'] = pd.qcut(data_st['prob'], g)
    
    ys = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).sum()
    yt = data_st['ViolentCrimesPerPop'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['prob'].groupby(data_st.dcl).sum()
    ypt = data_st['prob'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pval = 1 - chi2.cdf(hltest, g-2)
    
    df = g-2

    print('\n HL-chi2({}): {}, p-value: {}\n'.format(df, hltest, pval))

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
        result = result + np.dot(np.transpose(x_full[i, :]), 
                                 x_full[i, :]) * pb[i,1] * pb[i, 0]
    vcov = np.linalg.inv(np.matrix(result))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    pval = (1 - norm.cdf(abs(t))) * 2
    print(pd.DataFrame(pval, 
                       index=['intercept','population','pctUrban', 
                              'perCapInc', 'PctPopUnderPov'], 
                       columns=['p-value']))

# Create scores dictionary for each algorithm
def base_evaluator(X_train, y_train, X_test, y_test, models, metric="f1", average='macro'):
    mdl=[]
    results=[]
    for model in models.keys():
        print(f"Model: {model}")
        est = models[model]
        est.fit(X_train,  y_train)
        mdl.append(model)
        y_pred = est.predict(X_test)
        # will also need probabilistic predictions to evaluate!
        y_pred_prob = est.predict_proba(X_test)
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
        print(metric)
        print(alg)
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
                       evaluation_metric="f1", average='macro'):
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

def get_scorer(metric='f1', average='macro'):
    scorers = {
        'precision': make_scorer(precision_score, average=average),
        'recall': make_scorer(recall_score, average=average),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average=average),
        'auc': make_scorer(roc_auc_score, average=average)
    }
    return scorers[metric]