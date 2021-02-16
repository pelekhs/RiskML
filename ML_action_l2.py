# %%
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import grouped_tuning
from evaluation import grouped_evaluation
from preprocessing import preprocessor, ColToOneHot, merge_low_frequency_columns, load_datasets
from globals import *

LEVEL1_CATEGORY = "Malware"
RESULTS_DIR = "./results/"
INDUSTRY_ONE_HOT = True
JSON_DIR = '../VCDB/data/json/validated/'
CSV_DIR = "../csv/"
PREDICTORS = ["asset.variety", "asset.assets.variety", "victim.industry", "victim.orgsize"]
TRAIN_SIZE = 0.8
TARGETS = ['action.malware.variety.Ransomware',
           'action.malware.variety.Backdoor',
           'action.malware.variety.C2',
           "action.malware.variety.Spyware/Keylogger",
           "action.malware.variety.Capture app data"]
STRATIFY = "action.malware.variety.Ransomware"
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
    ## First filter dataset by level1 action
    df, veris_df = pp.filter_by_verisdf_col(column="action." + LEVEL1_CATEGORY)

    ## Feature selection
    dataset, veris_df = pp.add_predictors(predictors=PREDICTORS)

    # dataset = df.iloc[veris_df["index"]][["victim.industry",
    #                                       "victim.industry.name",
    #                                       "victim.orgsize",
    #                                       "asset.variety",
    #                                       "asset.assets.variety",
    #                                       ".".join(["action",
    #                                                 ACTION2.lower(),
    #                                                 "variety"])]] \
    #             .reset_index(drop=True)

    ##  Pipeline 2.1
    ### Imputation method 1: Don't impute - just drop
    dataset, veris_df = pp.imputer(method="dropnan")
    #dataset, veris_df = pp.use_unknown_class()
    # hacking_dataset = hacking_dataset.replace("?", np.nan)
    # na_free = hacking_dataset.dropna(how='any', axis=0)
    # ##### keep dropped rows and also drop them from veris_df
    # dropped_indices = hacking_dataset[~hacking_dataset.index.isin(na_free.index)].index
    # veris_df = veris_df.drop(dropped_indices).reset_index(drop=True)
    # hacking_dataset = na_free.reset_index(drop=True)

    #### Define multi-target dataset
    ys = ColToOneHot(collapsed=dataset,
                     veris_df=veris_df,
                     father_col="action." + LEVEL1_CATEGORY.lower() + ".variety",
                     replace=False)
    ys, dataset, veris_df = pp.process_target(ys, targets=TARGETS)
    # ##### Drop Unknown column (Not useful here)
    # ys = ys.drop(columns=".".join(["action", ACTION2.lower(), "variety", "Unknown"]))
    # ##### Merge varieties that have very few samples
    # ys = merge_low_frequency_columns(df=ys,
    #                                  column_regex_filter=(f"action\.hacking\.variety"),
    #                                  threshold=0.04,
    #                                  merge_into="action.hacking.variety.Other",
    #                                  drop_merged=True)
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
    FOLDER_NAME = "action.x.variety"
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

# [".".join(["action", ACTION2.lower(), "variety", "Other"])])
# action_features = df.filter(regex=("action\..*\.variety|^action$"), axis=1)

# %%