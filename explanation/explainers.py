from scipy.sparse.construct import rand
import shap
import lightgbm as lgb
from lightgbm import LGBMClassifier, plot_importance,plot_metric, plot_tree
import numpy as np

lgbm_default_params = {
    "max_bin": 255,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "verbose": -1,
    "min_data": 20,
    "boost_from_average": True
    }

def shap_lgbm(X_train, 
              X_test, 
              y_train,
              y_test):
    "Applies TreeExplainer that is appropriate for the LightGBM classifier"
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test)

    model_shap = lgb.train(params=lgbm_default_params,
                           train_set=d_train, 
                           num_boost_round=100, 
                           valid_sets=[d_test], 
                           early_stopping_rounds=50, 
                           verbose_eval=1000)
    explainer = shap.TreeExplainer(model_shap)
    shap_values = explainer.shap_values(X_test)

    return model_shap, shap_values, explainer

def shap_generic(X_train, 
                 X_test, 
                 y_train,
                 y_test, 
                 model):
    "Applies KernelExplainer (slow) to be able to explain any model"
    model_shap = model
    explainer = shap.KernelExplainer(model.predict, X_test)
    shap_values = explainer.shap_values(X_test)

    return model_shap, shap_values, explainer

def run_explainer(estimator, 
                  X_train, 
                  X_test, 
                  y_train, 
                  y_test, 
                  pipeline, 
                  data_percentage=0.05,
                  test_percentage=0.3):
    """ 
    This function uses SHAP subfunctions to produce explanations of the given estimator.
    Subsampling of the dataset is implemented to reduce execution times.
       
    Parameters
    ---------- 
        estimator: sklearn / LightGBM estimator class
            The model to be explained

        X_train: Pandas DataFrame
            Input train dataset

        y_train: Pandas Series
            Output train vector (binary targets)

        X_test: Pandas DataFrame
            Input test dataset

        y_test: Pandas Series
            Output test vector (binary targets)
        
        pipeline: sklearn.Pipeline object
            Pipeline of processing the dataset

        metric: str
            Metric to optimise. Choose from: "precision", "recall", "f1", 
            "hl", "accuracy", "auc"

        data_percentage: 0 < float < 1
            Dataset fraction to be used with SHAP for explanations

        test_over_train_percentage: 0 < float < 1
            Training set fraction to be used as test with SHAP for explanations
    ---------- 
    Returns:

    X_shap_test:  DataFrame
        Fraction (data_percentage) of the initial test dataset
    
    model_shap: sklearn / LightGBM estimator class object
        The model that is fit to the shap dataset (in case of lgbm model is refitted)
    """
    
    """  ML classifier hyperparameter tuning """
    import matplotlib.pyplot as plt

    N = int(X_train.shape[0] * data_percentage)
    N_test = int(X_train.shape[0] * data_percentage * test_percentage)
    
    rs = np.random.randint(0,100)
    X_shap_train = pipeline[:2].fit_transform(shap.utils.sample(X_train, N, random_state=rs))
    y_shap_train = shap.utils.sample(y_train, N, random_state=rs)

    X_shap_test = pipeline[:2].fit_transform(shap.utils.sample(X_test, N_test, random_state=rs))
    y_shap_test = shap.utils.sample(y_test, N_test, random_state=rs)

    if isinstance(estimator, LGBMClassifier):
        
        tree = plt.figure()
        plot_tree(estimator,tree_index=0)
        tree.savefig("explain_plots/tree_plot.png")

        importance = plt.figure()
        plot_importance(booster=estimator)
        importance.savefig("explain_plots/importance_plot.png")                
        
        model_shap, shap_values, _= \
            shap_lgbm(X_shap_train, 
                      X_shap_test, 
                      y_shap_train, 
                      y_shap_test)
    else:
        model = pipeline[-1]
        model_shap, shap_values, _ = \
            shap_generic(X_shap_train, 
                         X_shap_test, 
                         y_shap_train, 
                         y_shap_test, 
                         model) 

    # force = plt.figure()
    # shap.force_plot(explainer.expected_value, 
    #                 shap_values,
    #                 X_shap_test)
    # force.savefig("explain_plots/force_plot.png")

    dependence = plt.figure()
    shap.dependence_plot("victim.orgsize.Large", shap_values[0], X_shap_test)
    dependence.savefig("explain_plots/dependence_plot_orgsize0.png")
    dependence.show()

    dependence = plt.figure()
    shap.dependence_plot("victim.orgsize.Large", shap_values[1], X_shap_test)
    dependence.savefig("explain_plots/dependence_plot_orgsize1.png")
    dependence.show()
    
    dependence = plt.figure()
    shap.dependence_plot("victim.industry.name.Healthcare", shap_values[0], X_shap_test)
    dependence.savefig("explain_plots/dependence_plot_industry_name0.png")
    dependence.show()
    
    dependence = plt.figure()
    shap.dependence_plot("victim.industry.name.Healthcare", shap_values[1], X_shap_test)
    dependence.savefig("explain_plots/dependence_plot_industry_name1.png")
    dependence.show()

    summary = plt.figure()
    shap.summary_plot(shap_values, X_shap_test)
    summary.savefig("explain_plots/summary_plot.png")
    summary.show()

    #shap.plots.waterfall(shap_values[0], max_display=20)
    return X_shap_test, model_shap

if __name__ == "__main__":
    run_explainer()

