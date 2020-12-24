import os
import pandas as pd
from verispy import VERIS
from create_csv import create_veris_csv
import argparse
import sys

# from create_csv import create_veris_csv

JSON_DIR = '../VCDB/data/json/validated/'

CSV_DIR = "../csv/"

COLUMNS_TO_DROP_BY_TERM = ["cve",
                           "asset.cloud",
                           "notes",
                           "incident_id",
                           "plus"]

COLUMNS_TO_DROP = ["pattern",
                   "actor.partner.country",
                   "actor.external.country",
                   "action.unknown.result",
                   "asset.hosting",
                   "attribute.confidentiality.state",
                   "campaign_id",
                   "victim.revenue.iso_currency_code",
                   "victim.secondary.amount",
                   "victim.secondary.amount"]

COLUMNS_TO_MANAGE_TIME = ["attribute.availability.duration",
                          "timeline.compromise",
                          "timeline.discovery"]

RECREATE_VERIS = True

NEW_NAN = "NA"

NEW_UNKNOWN_VALUE = "?"


def parse_arguments ():
    parser = argparse.ArgumentParser(
        description="Risk assessment preprocessor"
    )
    # we need to give the name of X file, and the y file as well.
    # with alpha, it makes three parameters.
    parser.add_argument("-j",
                        "--json_dir",
                        help="the directory containing the json files",
                        default=JSON_DIR)
    parser.add_argument("-c",
                        "--csv_dir",
                        help="the directory containing the csv files",
                        default=CSV_DIR)
    parser.add_argument("-dt",
                        "--columns_to_drop_by_term",
                        action='append',
                        help="terms (list like) that will cause any column containing them to be dropped",
                        default=COLUMNS_TO_DROP_BY_TERM)
    parser.add_argument("-d",
                        "--columns_to_drop",
                        action='append',
                        help="columns names (list like) to directly drop",
                        default=COLUMNS_TO_DROP)
    parser.add_argument("-r",
                        "--recreate_veris",
                        type=bool,
                        help="if to produce or not the veris format csv",
                        default=RECREATE_VERIS)
    return parser.parse_args()

def subset (df, feat_columns, target_columns):
    features = df[feat_columns]
    output = df[target_columns]
    return features, output

class DataFrameHandler:
    def __init__(self, veris_df, collapsed, new_unknown_value=NEW_UNKNOWN_VALUE, new_nan=NEW_NAN):
        """

        Parameters
        ----------
        new_unknown_value : object
        """
        self.veris_df = veris_df
        self.collapsed = collapsed
        self.new_unknown_value = new_unknown_value
        self.new_nan = new_nan


    def replace_with (self, columns, to_replace='Unknown'):
        #self.collapsed = self.collapsed_ if inplace else self.collapsed_.copy()
        for column in columns:
            self.collapsed[self.collapsed[column] == to_replace] = self.new_unknown_value
        return self.collapsed

    def drop_columns_that_contain_term (self, terms='Unknown'):
        for term in list(terms):
            # regex = contains
            self.collapsed.drop(columns=self.collapsed.filter(regex=(r'\b'+ term +r'\b'), axis=1).columns,
                                inplace=True)
        return self.collapsed

    def manage_unknown (self, fathers):
        """ This method replaces Unknown with a selected replacement value (e.g "?") for each child column
        that is relevant to a specific label of each father column. This is needed to show that such Unknowns truly
        are missing data and are not empty due to lack of activation of a father column. """
        def get_cols_of_true_unknowns(veris_row):
            for father in list(fathers):
                labels = set(collapsed[father].unique())
                labels.remove('Unknown')
                labels.remove('Multiple')
                for label in labels:
                    Child = ".".join([father, label])
                    child = Child.lower()
                    # locate grandchildren (length=4) that finish with .Unknown
                    unknown_grandchildren = [c for c in self.veris_df.columns
                                             if c.startswith(child)
                                             and c.endswith("Unknown")
                                             and len(c.split(".")) == 4]
                    # case 1: father.Label is False -> all grandchildren are Nan and not Unknown (so no need to change)
                    # case 2: child.Unknown = 1 while Father is pointing at it -> Child is really missing
                    if veris_row[Child] == 1:
                        true_unknown_vector = []
                        for grandchild in unknown_grandchildren:
                            # children with 1 are true Unknowns
                            if veris_row[grandchild] == 1:
                                # store grandchild for the row ignoring "Unknown at the end"
                                true_unknown_vector.append(".".join((grandchild.split(".")[:-1])))
                        return true_unknown_vector

        # get dataframe that lists all columns that contain true unknown for every row
        true_unknown_df = self.veris_df.apply(lambda x: get_cols_of_true_unknowns(veris_row=x), axis=1)

        # replace True Unknowns with "?" in collapsed
        for index, _ in collapsed.iterrows():
            if true_unknown_df.iloc[index]:
                self.collapsed.loc[index, true_unknown_df.iloc[index]] = self.new_unknown_value
        # columns that need to replace Fake Unknowns with Nans
        changed_cols = list(self.collapsed.filter(regex=(r'\baction\.\b')).columns) + \
                       list(self.collapsed.filter(regex=(r'\bactor\.\b')).columns)
        # Replace fake Unknowns with NA
        self.collapsed.loc[:][changed_cols] = self.collapsed.loc[:][changed_cols].replace('Unknown',
                                                                                          self.new_nan)
        return self.collapsed

    # def unify_results(self):
    #     """ This function creates 3 One Hot columns
    #     (action.result.elevate, action.result.infiltrate, action.result.exfiltrate)
    #     concerning the action result and drops all the others"
    #     """
    #
    #     def unify(input_row): # input row is a pd.Series()
    #         return sum(input_row) > 0
    #
    #     #self.collapsed = self.collapsed_ if inplace else self.collapsed_.copy()
    #
    #     infiltrate_cols = df_columns_that_contain_term("result.Infiltrate", veris_df)
    #     exfiltrate_cols = df_columns_that_contain_term("result.Exfiltrate", veris_df)
    #     elevate_cols = df_columns_that_contain_term("result.Elevate", veris_df)
    #
    #     """
    #     Exfiltrate is True if the sum of labels (0,1) of all columns relevant to Exfiltrate (columns that contain the
    #     term "result.Exfiltrate") is > 0. This means that at least one exfiltrate column is True
    #     (originating from a particular action) and this is enough to decide given that actions are mutually exlusive. (NOT!)
    #     The only problem according to the dataset is when an action has MULTIPLE results and this method takes that
    #     into account. Same for the other 2 results.
    #     """
    #     self.collapsed["action.result.exfiltrate"] = veris_df.apply(lambda x: unify(input_row=x[infiltrate_cols]),
    #                                                                 axis=1)
    #     self.collapsed["action.result.infiltrate"] = veris_df.apply(lambda x: unify(input_row=x[exfiltrate_cols]),
    #                                                                 axis=1)
    #     self.collapsed["action.result.elevate"] = veris_df.apply(lambda x: unify(input_row=x[elevate_cols]),
    #                                                               axis=1)
    #     # drop initial result columns
    #     self.collapsed.drop(columns=df_columns_that_end_with(".result", self.collapsed),
    #                         inplace=True)
    #     return self.collapsed

    def manage_malware_name_nans (self):

        def replace_malware_nan (action_malware, malware_name):
            if not(isinstance(malware_name, str)):
                if action_malware == 1:
                    malware_name = self.new_unknown_value
                else:
                    malware_name = self.new_nan
            return malware_name
        self.collapsed["action.malware.name"] = \
            self.veris_df.apply(lambda x: replace_malware_nan(x["action.Malware"],
                                                               x["action.malware.name"]),
                                axis=1)
        return self.collapsed

    def convert_time (self, columns, drop=True):

        def to_seconds(unit, value, replace_unknown='?'):
            mapper = {'Seconds': 1,
                      'Minutes': 60,
                      'Hours': 3600,
                      'Days': 86400,
                      'Weeks': 604800,
                      'Months': 2592000,
                      'Years': 31536000
                      }
            if unit == 'Seconds':
                return mapper['Seconds'] * value
            if unit == 'Minutes':
                return mapper['Minutes'] * value
            if unit == 'Hours':
                return mapper['Hours'] * value
            if unit == 'Days':
                return mapper['Days'] * value
            if unit == 'Weeks':
                return mapper['Weeks'] * value
            if unit == 'Months':
                return mapper['Months'] * value
            if unit == 'Years':
                return mapper['Years'] * value
            else:
                return replace_unknown

        for column in columns:
            self.collapsed[column] = self.collapsed.apply(lambda x: to_seconds(unit=x[column + '.unit'],
                                                                               value=x[column + '.value'],
                                                                               replace_unknown=self.new_unknown_value),
                                                          axis=1)
            if drop:
                self.collapsed.drop(columns=[column + '.unit', column + '.value'],
                                    inplace=True)
        return self.collapsed

if __name__ == "__main__":

    args = parse_arguments()

    # LOAD csv datasets from csv_dir
    if args.recreate_veris:
        create_veris_csv(args.json_dir, args.csv_dir, "veris_df.csv")
    v = VERIS(json_dir=args.json_dir)
    veris_df = pd.read_csv(os.path.join(args.csv_dir, "veris_df.csv"),
                           index_col=0,
                           low_memory=False)

    collapsed = pd.read_csv(os.path.join(args.csv_dir, "Rcollapsed.csv"),
                            sep=",",
                            encoding='utf-8',
                            index_col=0,
                            low_memory=False) \
                    .reset_index(drop=True)

    etl = collapsed.copy()

    # Dataset Curation

    # filter out environmental incidents
    collapsed = collapsed[collapsed["action"] != "Environmental"]

    # ETL using the handler

    # Locate real and fake Unknown values (choose "?" symbol as new "Unknown")
    dfh = DataFrameHandler(veris_df=veris_df, collapsed=etl, new_unknown_value=NEW_UNKNOWN_VALUE)
    # Manage Unknown
    etl = dfh.manage_unknown(fathers=['action', 'actor'])
    etl = dfh.manage_malware_name_nans()
    # Manage time (create seconds column where multiple columns are used for timelines)
    etl = dfh.convert_time(columns=COLUMNS_TO_MANAGE_TIME)
    # Drop columns that contain term
    etl = dfh.drop_columns_that_contain_term(terms=args.columns_to_drop_by_term)


    # Manual ETL

    # Manually drop columns
    print("The followind columns will be explicitly dropped:\n{}\n".
          format(COLUMNS_TO_DROP))
    etl = etl.drop(columns=COLUMNS_TO_DROP)
    # Manually replace remaining nans and Unknowns in a meaningful manner
    # motive
    motive_cols = etl.filter(regex=(r'\bmotive\b'), axis=1).columns
    etl[motive_cols] = etl[motive_cols].fillna("No motive")
    # action.malware.variety (only one case)
    etl["action.malware.variety"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # action.misuse.vector (only one case)
    etl["action.misuse.vector"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # action.social.vector (only one case)
    etl["action.social.vector"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # actor.external.variety (don't know why but all of them come from external actors so ?)
    etl["actor.external.variety"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # actor.internal.variety (same as above)
    etl["actor.internal.variety"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # assets
    etl["asset.assets.variety"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["asset.variety"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    # misc
    etl["victim.victim_id"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["victim.country"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["reference"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["timeline.containment.unit"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["timeline.incident.month"].fillna(NEW_UNKNOWN_VALUE, inplace=True)
    etl["pattern_collapsed"].fillna(NEW_UNKNOWN_VALUE, inplace=True)

    # what remains as Unknown is considered as truly Unknown -> "?"
    print("The followind columns will explicitly have Unknown replaced with {}:\n{}\n".
          format(NEW_UNKNOWN_VALUE, etl.columns[(etl.values == 'Unknown').any(0)].tolist()))
    etl.replace('Unknown', NEW_UNKNOWN_VALUE, inplace=True)
    etl.replace('unknown', NEW_UNKNOWN_VALUE, inplace=True)

    # To CSV
    save_as = os.path.join(args.csv_dir, "etl_product.csv")
    etl.to_csv(save_as)

    print("Dataset stored as: {}".format(save_as))

