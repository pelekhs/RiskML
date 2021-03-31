from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as classmetrics
from collections.abc import Iterable
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer
from sklearn.metrics import recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2, norm


# Hosmer Lemeshow
# See https://jbhender.github.io/Stats506/F18/GP/Group13.html
def hl_test(y_true, y_pred, g=10, threshold=0.5):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data
    Input: dataframe(data), integer(num of subgroups divided)
    Output: float
    '''
    data = pd.DataFrame()
    data['true'] = y_true
    data['prob'] = y_pred

    data.loc[data['true'] > threshold, 'true'] = 1
    data.loc[data['true'] <= threshold, 'true'] = 0

    data_st = data.sort_values('prob')
    
    try:
        data_st['dcl'] = pd.qcut(data_st['prob'], g)
    except ValueError:
        return 1
    
    ys = data_st['true'].groupby(data_st.dcl).sum()
    yt = data_st['true'].groupby(data_st.dcl).count()
    yn = yt - ys
    
    yps = data_st['prob'].groupby(data_st.dcl).sum()
    ypt = data_st['prob'].groupby(data_st.dcl).count()
    ypn = ypt - yps
    
    hltest = ( ((ys - yps)**2 / yps) + ((yn - ypn)**2 / ypn) ).sum()
    pval = 1 - chi2.cdf(hltest, g-2)

    return pval

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


def get_scorer(metric=None, average='macro'):
    scorers = {
        'precision': make_scorer(precision_score, average=average),
        'recall': make_scorer(recall_score, average=average),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average=average),
        'auc': make_scorer(roc_auc_score, average=average),
        'hl': make_scorer(hl_test, needs_proba=True)
    }
    if metric in ['precision', 'recall', 'accuracy', 'f1', 'auc', 'hl']:
        return scorers[metric]
    elif metric=='all':
        metrics = ['precision', 'recall', 'f1', 'auc']
        averaging = ['macro', 'micro', 'weighted']
        
        scorer_names = ['-'.join([m, a]) for m in metrics for a in averaging]
        scorers = [get_scorer(m, a) for m in metrics for a in averaging] 
        scoring = dict(zip(scorer_names, scorers))
        
        scoring['accuracy'] = get_scorer('accuracy')
        scoring['hl'] =  get_scorer('hl')
        return scoring
    elif metric=='gridsearch':
        # return all metrix with the specific averaging type
        return scorers
    else:
        print("Error: Please correctly define metric. Cannot get scorer!")

def evaluate_cv(estimator, X, y, n_folds=5, random_state=None):
    
    # Get dictionary of scorers
    scoring = get_scorer(metric='all')
    
    # Define CV
    skf = StratifiedKFold(n_splits=n_folds, 
                          shuffle=True if isinstance(random_state, int) else False, 
                          random_state=random_state)
    # Cross validate
    cv_results = cross_validate(estimator=estimator, 
                                X=X, 
                                y=y, 
                                scoring=scoring, 
                                cv=skf)

    # Get mean of folds performance
    metrix_dict = {k: np.mean(v) for k, v in cv_results.items()}

    return metrix_dict