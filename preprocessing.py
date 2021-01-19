from etl import JSON_DIR, CSV_DIR
from verispy import VERIS
import pandas as pd
import os
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

v = VERIS(json_dir=JSON_DIR)

def load_datasets():
    veris_df = pd.read_csv(os.path.join(CSV_DIR, "veris_df.csv"),
                           index_col=0,
                           low_memory=False)

    df = pd.read_csv(os.path.join(CSV_DIR, "etl_product.csv"),
                     low_memory=False,
                     index_col=0)
    return df, veris_df

def mapper(x, asset_variety):
    splitted = x.split(" - ")
    splitted[0] = f"asset.assets.variety.{asset_variety}"
    return ".".join(splitted)

def ColToOneHot(collapsed, veris_df, father_col="action", replace=True) -> object:
    if father_col.startswith("asset.assets.variety."):
        asset_variety = father_col.split(".")[-1]
        columns = veris_df.filter(like=f"asset.assets.variety.{asset_variety[0]} -").columns
        renamed_columns = columns.map(lambda x: mapper(x, asset_variety))
        renamer = dict(zip(columns.tolist(), renamed_columns.tolist()))
        veris_df.rename(mapper=renamer,
                        axis="columns",
                        inplace=True)
    if replace:
        collapsed_ = collapsed.copy()
        for attr in list(v.enum_summary(veris_df, father_col).iloc[:, 0]):
            sub = father_col + "." + attr
            collapsed_[sub] = veris_df[sub].astype(int)
        collapsed_.drop(columns=father_col, inplace=True)
        return collapsed_
    else:
        OneHot = pd.DataFrame()
        for attr in list(v.enum_summary(veris_df, father_col).iloc[:, 0]):
            sub = father_col + "." + attr
            OneHot[sub] = veris_df[sub]
        return OneHot.astype(int)

# Nee to evolve into a class with methods
class preprocessor():

    def __init__ (self, df, veris_df):
        self.df = df
        self.veris_df = veris_df

    def drop_environmental(self):
        """
        df: The dataset product of the etl procedure [pandas dataframe]
        veris_df: the corresponding veris dataframe [pandas dataframe]
        """
        # Drop environmental for both datasets. (Veris_df is essential for one hot encoding later)
        ## Drop rows
        self.df = self.df.drop(self.veris_df[self.veris_df["action.Environmental"]].index).reset_index(drop=True)
        self.veris_df = self.veris_df.drop(self.veris_df[self.veris_df["action.Environmental"]].index) \
            .reset_index(drop=True)
        ## Drop cols
        self.df = self.df.drop(columns=self.df.filter(like="environmental").columns)
        self.veris_df = self.veris_df.drop(columns=self.veris_df.filter(like="environmental").columns)
        return self.df, self.veris_df

    def round_NAICS(self, digits=2):
        ## Keep only 2 digits of NAICS
        self.df['victim.industry'] = self.df['victim.industry'].apply(lambda x: str(x)[:digits])
        # also need to merge columns in veris_df
        return self.df

    def imputer(self, method="dropnan"):
        if method == "dropnan":
            self.simple_nan_dropper()
        if method == "create unknown class":
            pass
        return self.df, self.veris_df

    def simple_nan_dropper(self):
        """ This function just replaces ? with nans and drops rows containing even one. It also stores
        the deleted indices and drops them from veris_df as well"""
        self.df = self.df.replace(np.nan, "NA")
        self.df = self.df.replace("?", np.nan)
        na_free = self.df.dropna(how='any', axis=0)
        ##### keep dropped rows and also drop them from veris_df
        dropped_rows = self.df[~self.df.index.isin(na_free.index)].index
        self.veris_df = self.veris_df.drop(dropped_rows)
        self.df = na_free
        return self.df, self.veris_df

    def filter_by_verisdf_col(self, column):
        """ This function gets a column name and filters dataset according to the column.
        Only entries where this column has a True value are kept
        """
        #filter
        self.veris_df = self.veris_df[self.veris_df[column] == True]
        # syncronise df with veris_df
        self.df = self.df.iloc[self.veris_df.index, :]
        return self.df, self.veris_df

    def add_predictors(self, predictors):
        dataset = pd.DataFrame(index=self.df.index)
        # predictors only for action prediction
        if "asset.variety" in predictors:
            dataset = pd.concat([dataset, self.df["asset.variety"]],
                                join="inner", axis=1)
        if "asset.assets.variety" in predictors:
            dataset = pd.concat([dataset, self.df["asset.assets.variety"]],
                                join="inner", axis=1)
        # predictors only for asset prediction
        if "action" in predictors:
            dataset = pd.concat([dataset, self.df["action"]],
                                join="inner", axis=1)
        if "action.x.variety" in predictors:
            action_x_variety_columns = self.df.filter(regex="action\..*\.variety").columns.tolist()
            dataset = pd.concat([dataset, self.df[action_x_variety_columns]],
                                join="inner", axis=1)
        # standard predictors
        if "victim.orgsize" in predictors:
            dataset = pd.concat([dataset, self.df["victim.orgsize"]],
                                join="inner", axis=1)
        if "victim.industry" in predictors:
            dataset = pd.concat([dataset, self.df[["victim.industry", "victim.industry.name"]]],
                                join="inner", axis=1)
        self.df = dataset
        return self.df, self.veris_df

    def process_target(self, ys, targets):
        """ Default preprocessing and target selection method for multioutput target datasets"""
        self.ys = ys
        # locate rows to drop
        rows_to_drop = np.ones(self.ys.shape[0]).astype(bool)
        for target in targets:
            # only delete rows that have values diff than 0,1 for the targets of interest
            local_rows_to_drop_1 = (self.ys[target] != 0).values
            local_rows_to_drop_2 = (self.ys[target] != 1).values
            local_rows_to_drop = np.bitwise_and(local_rows_to_drop_1, local_rows_to_drop_2)
            rows_to_drop = np.bitwise_and(local_rows_to_drop, rows_to_drop)
        rows_to_drop = np.where(rows_to_drop)[0].reshape(-1).tolist()
        # Delete selected rows from all datasets
        self.ys.drop(rows_to_drop, inplace=True)
        self.ys.reset_index(drop=True, inplace=True)
        self.df.drop(rows_to_drop, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.veris_df.drop(rows_to_drop, inplace=True)
        self.veris_df.reset_index(drop=True, inplace=True)
        # Drop useless cols
        cols_to_drop = list(set(ys.columns) - set(targets))
        ys.drop(columns=cols_to_drop, inplace=True)
        # only applies to assets
        if "asset.variety.Kiosk/Term" in targets:
            self.ys.rename(mapper={"asset.variety.Kiosk/Term": "asset.variety.Kiosk-Term"},
                           axis="columns",
                           inplace=True)
        if "action.malware.variety.Backdoor" in targets and "action.malware.variety.C2" in targets:
            self.ys["action.malware.variety.Backdoor or C2"] = \
                self.ys.apply(lambda x: x["action.malware.variety.C2"] or x["action.malware.variety.Backdoor"], axis=1)
            self.ys.drop(columns=["action.malware.variety.Backdoor", "action.malware.variety.C2"], inplace=True)
        return self.ys, self.df, self.veris_df

    def one_hot_encode_and_scale(self, predictors, ind_one_hot=True):
        # victim.orgsize
        if "victim.orgsize" in predictors:
            self.df = ColToOneHot(self.df, self.veris_df, father_col="victim.orgsize")
            # drop dummy
            self.df.drop(columns=["victim.orgsize.Small"], inplace=True)
        # action
        if "action" in predictors:
            self.df = ColToOneHot(self.df, self.veris_df, father_col="action")
            self.df = self.df.drop(columns=["action.Unknown"])

        # action.x.variety
        if "action.x.variety" in predictors:
            lb = LabelBinarizer()
            for x in ["misuse", "physical", "error", "hacking", "social", "malware"]:
                self.df = ColToOneHot(self.df, self.veris_df, father_col=f"action.{x}.variety", replace=True)
                self.df.drop(columns=f"action.{x}.variety.Unknown", inplace=True)
        # assets
        if "asset.variety" in predictors:
            self.df = ColToOneHot(self.df, self.veris_df, father_col="asset.variety")
            self.df = self.df.drop(columns=["asset.variety.Unknown"])
        if "asset.assets.variety" in predictors:
            self.df = ColToOneHot(self.df, self.veris_df, father_col="asset.assets.variety")
            self.df = self.df.drop(columns=["asset.assets.variety.Unknown"])
        # victim.industry
        if "victim.industry" in predictors:
            if ind_one_hot:
                self.NAICS = "You have requested one hot encoding so NAICS has no value"
                lb = LabelBinarizer()
                self.industry_one_hot = pd.DataFrame(data=lb.fit_transform(self.df["victim.industry"]),
                                                columns=["victim.industry." + no for no in lb.classes_])
                # drop dummy
                self.industry_one_hot = self.industry_one_hot.iloc[:, :-1]
                self.df.drop(columns=["victim.industry", "victim.industry.name"], inplace=True)
                self.df = pd.concat([self.df, self.industry_one_hot], axis=1)
            else:
                sc = StandardScaler()
                self.NAICS = sc.fit_transform(df["victim.industry"].values.reshape(-1, 1))
                self.df["victim.industry"] = NAICS
        self.df = self.df.astype(int).reset_index(drop=True)
        # if "victim.orgsize" not in predictors:
        #     X.drop(columns=["victim.orgsize.Large"], inplace=True)
        return self.df, self.NAICS, self.industry_one_hot


def merge_low_frequency_columns (df,
                                 column_regex_filter,
                                 threshold=0.05,
                                 merge_into="action.hacking.variety.Other",
                                 drop_merged=True):
    """
    This function calculates the number of trues (if columns is boolean) or ones (if column is int) in each candidate
    column of the dataframe, calculates the ratio with the total of true values for all candidate columns
    and if this ratio is smaller than the threshold it merges all of these columns into a new column that is defined
    by the "merge_into" argument. Arguments:

    df: dataframe whose boolean or int columns need to be merged based on their total cardinality
    column_regex_filter: regex expression to set the candidate columns of the dataframe (columns need to be int or bool)
    threshold: the value of the ratio of column_sum/total_sum below which a column is chosen for merging
    merge_into: the name of the column to hold the merge results
    drop: if True drop the merged columns
    """
    # subset df based on required cols
    df_ =df.copy()
    # subset of df holding only the candidate columns
    candidate_df = df_.filter(regex=column_regex_filter, axis=1)
    frequencies = candidate_df.sum()
    total_sum = frequencies.sum()
    # candidate columns to be merged due to few samples
    chosen_cols = frequencies[frequencies / total_sum < threshold].index.tolist()
    # print(chosen_cols)
    print(candidate_df[chosen_cols])
    # just check if there is a 1 in the columns that need to be merged to a new one
    merge = lambda x: 0 if sum(x) == 0 else 1
    # print(candidate_df.columns)
    print(candidate_df[chosen_cols])
    if drop_merged:
        df_.drop(columns=chosen_cols, inplace=True)
    df_[merge_into] = candidate_df.apply(lambda x: merge(x[chosen_cols]), axis=1)
    return df_

def create_log_folder(pipeline, results_type="Tuning", results_root=os.curdir):
    now = datetime.datetime.now() + datetime.timedelta()
    pipeline_dir = os.path.join(results_root, pipeline, str(now.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(pipeline_dir)
    save_dir = os.path.join(pipeline_dir, results_type)
    os.makedirs(save_dir)
    return save_dir


