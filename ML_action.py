from sklearn.model_selection import train_test_split
from hyperparameter_tuning import grouped_tuning
from evaluation import grouped_evaluation
from preprocessing import preprocessor, ColToOneHot, load_datasets
from globals import *

# Arguments
TUNE_METRIC = None
N_JOBS_CV = 6
EVALUATION_METRIC = 'f1'
RESULTS_DIR = "./results/"
INDUSTRY_ONE_HOT = True
JSON_DIR = '../VCDB/data/json/validated/'
CSV_DIR = "../csv/"
PREDICTORS = ["asset.variety", "asset.assets.variety", "victim.industry", "victim.orgsize"]
TARGETS = ["action.Hacking", "action.Malware", "action.Error", "action.Misuse",
           "action.Physical", "action.Social"]
STRATIFY = "action.Hacking"
TRAIN_SIZE = 0.8

if __name__ == "__main__":

    # Load data
    df, veris_df = load_datasets()

    # Data preprocessing = Feature selection, Imputation, One Hots etc...
    ## Drop environmental and modify NAICS to 2 digits
    pp = preprocessor(df, veris_df)
    df, veris_df = pp.drop_environmental()
    df = pp.round_NAICS(digits=2)

    # Pipeline 1: Predict 1st level action
    ## Feature selection
    dataset, veris_df = pp.add_predictors(predictors=PREDICTORS)

    ###  Pipeline 1.1
    #### Imputation method 1: Don't impute - just drop
    dataset, veris_df = pp.imputer(method="dropnan")

    #### Define targets
    ys = ColToOneHot(collapsed=dataset,
                     veris_df=veris_df,
                     father_col="action",
                     replace=False)
    ys, dataset, veris_df = pp.process_target(ys, targets=TARGETS)

    #### One Hot Encoding and Scaling
    X, NAICS, industry_one_hot = pp.one_hot_encode_and_scale(predictors=PREDICTORS,
                                                             ind_one_hot=INDUSTRY_ONE_HOT)
    X.drop(columns=X.filter(like="Unknown").columns, inplace=True)

    ## Train / Test split
    stratifier = STRATIFY if len(TARGETS) != 1 else TARGETS[0]
    X_train, X_test, y_trains, y_tests = \
        train_test_split(X, ys, train_size=TRAIN_SIZE,
                         test_size=1-TRAIN_SIZE,
                         shuffle=True,
                         stratify=ys[stratifier])

    ## Hyperparameter Tuning for training separate classifiers for each 2nd level action
    TUNE_METRIC = ""
    TUNE_AVERAGING = "macro"
    FOLDER_NAME = "action"
    N_JOBS_CV = 6


    if TUNE_METRIC in ["accuracy", "precision", "recall", "f1"]:
        tune_scores, tune_params = grouped_tuning(X_train, X_test, y_trains, y_tests,
                                                  results_dir=RESULTS_DIR, pipeline=FOLDER_NAME,
                                                  tune_metric=TUNE_METRIC, param_grid=PARAM_GRID,
                                                  n_jobs_cv=N_JOBS_CV,
                                                  average=TUNE_AVERAGING)

    ## Evaluation
    EVALUATION_METRIC = "f1"
    EVALUATION_AVERAGING = "macro"
    if EVALUATION_METRIC in ["accuracy", "precision", "recall", "f1"]:
        eval_scores = grouped_evaluation(X_train, X_test, y_trains, y_tests,
                                         evaluation_metric=EVALUATION_METRIC,
                                         average=EVALUATION_AVERAGING,
                                         models=MODELS)
    # Replace nan with meaningful values.
    # keep action.<>.variety or action
    # action_features = df.filter(regex=("action\..*\.variety|^action$"), axis=1)
    # ... Within ML pipelines


    # # NEW industry name to reduced name vector conversion:
    # industry_mapper = {
    #   "Retail ": "Retail",
    #   "Information ": "Information",
    #   "Finance ": "Finance",
    #   "Educational ": "Educational",
    #   "Healthcare ": "Healthcare",
    #   "Public ": "Public",
    #   "Agriculture ": "Other",
    #   "Mining ": "Other",
    #   "Utilities ": "Other",
    #   "Accomodation ": "Other",
    #   "Entertainment ": "Other",
    #   "Professional ": "Other",
    #   "Real Estate ": "Other",
    #   "Administrative ": "Other",
    #   "Management ": "Other",
    #   "Construction ": "Other",
    #   "Manufacturing ": "Other",
    #   "Transportation ": "Other",
    #   "Trade ": "Other",
    #   "?": "?"
    # }