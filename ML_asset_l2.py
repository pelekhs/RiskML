from sklearn.model_selection import train_test_split
from hyperparameter_tuning import grouped_tuning
from evaluation import grouped_evaluation
from preprocessing import preprocessor, ColToOneHot, merge_low_frequency_columns, load_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from globals import *

LEVEL1_CATEGORY = "Server"
RESULTS_DIR = "./results/"
INDUSTRY_ONE_HOT = True
JSON_DIR = '../VCDB/data/json/validated/'
CSV_DIR = "../csv/"
PREDICTORS = ["action", "action.x.variety", "victim.industry", "victim.orgsize"]
TRAIN_SIZE = 0.8
TARGETS = ['asset.assets.variety.Server.Database']
       #'asset.assets.variety.Server.Web application',
       #'asset.assets.variety.Server.Mail', 'asset.assets.variety.Server.File',
       #'asset.assets.variety.Server.Code repository']
       # 'asset.assets.variety.Server.Other',
       # 'asset.assets.variety.Server.POS controller',
       # 'asset.assets.variety.Server.DNS', 'asset.assets.variety.Server.Backup',
       # 'asset.assets.variety.Server.Remote access',
       # 'asset.assets.variety.Server.Mainframe',
       # 'asset.assets.variety.Server.Authentication',
       # 'asset.assets.variety.Server.Print',
       # 'asset.assets.variety.Server.Directory',
       # 'asset.assets.variety.Server.Payment switch',
       # 'asset.assets.variety.Server.ICS', 'asset.assets.variety.Server.Proxy',
       # 'asset.assets.variety.Server.DHCP', 'asset.assets.variety.Server.DCS',
       # 'asset.assets.variety.Server.Configuration or patch management',
       # 'asset.assets.variety.Server.VM host',
       # 'asset.assets.variety.Server.Log',
       # 'asset.assets.variety.Server.Unknown']

STRATIFY = 'asset.assets.variety.Server.Database'
#"action.hacking.variety.Use of stolen creds", "action.hacking.variety.Use of backdoor or C2",
#           "action.hacking.variety.DoS", "action.hacking.variety.Brute force"

if __name__ == "__main__":

    # Load data
    df, veris_df = load_datasets()

    # Data preprocessing = Feature selection, Imputation, One Hots etc...
    ## Drop environmental and modify NAICS to 2 digits
    pp = preprocessor(df, veris_df)
    df, veris_df = pp.drop_environmental()
    df = pp.round_NAICS(digits=2)

    # Pipeline 2: Predict action.x.variety for each one of the actions malware and hacking
    ## Only keep entries relevant to the action from both datasets
    df, veris_df = pp.filter_by_verisdf_col(column="asset.variety." + LEVEL1_CATEGORY)

    ## Feature selection
    dataset, veris_df = pp.add_predictors(predictors=PREDICTORS)

    ##  Pipeline 2.1
    ### Imputation method 1: Don't impute - just drop
    dataset, veris_df = pp.imputer(method="dropnan")

    #### Define multi-target dataset
    ys = ColToOneHot(collapsed=dataset,
                     veris_df=veris_df,
                     father_col=f"asset.assets.variety.{LEVEL1_CATEGORY}",
                     replace=False)
    ys, dataset, veris_df = pp.process_target(ys, targets=TARGETS)

    #### One Hot Encoding and Scaling
    X, NAICS, industry_one_hot = pp.one_hot_encode_and_scale(predictors=PREDICTORS,
                                                             ind_one_hot=INDUSTRY_ONE_HOT)

    ## Train/ Test split
    stratifier = STRATIFY if len(TARGETS) != 1 else TARGETS[0]
    X_train, X_test, y_trains, y_tests = \
        train_test_split(X, ys, train_size=TRAIN_SIZE,
                         test_size=1-TRAIN_SIZE,
                         shuffle=True,
                         stratify=ys[stratifier])

    ## Hyperparameter Tuning for training separate classifiers for each 2nd level action
    TUNE_METRIC = ""
    TUNE_AVERAGING = "macro"
    N_JOBS_CV = 6
    FOLDER_NAME = "asset.assets.variety.x"
    # if TUNE_METRIC in ["accuracy", "precision", "recall", "f1"]:
    #     tune_scores, tune_params = grouped_tuning(X_train, X_test, y_trains, y_tests,
    #                                               results_dir=RESULTS_DIR, pipeline=FOLDER_NAME,
    #                                               tune_metric=TUNE_METRIC, param_grid=PARAM_GRID,
    #                                               n_jobs_cv=N_JOBS_CV,
    #                                               average=TUNE_AVERAGING)
    #
    ## Evaluation
    EVALUATION_METRIC = "f1"
    EVALUATION_AVERAGING = "macro"
    if EVALUATION_METRIC in ["accuracy", "precision", "recall", "f1"]:
        eval_scores = grouped_evaluation(X_train, X_test, y_trains, y_tests,
                                         evaluation_metric=EVALUATION_METRIC,
                                         average=EVALUATION_AVERAGING,
                                         models=MODELS)
