import pandas as pd
import os
import datetime
import numpy as np
import re
import logging 
import itertools

# Need to evolve into a class with methods
class etl_maker():

    def drop_environmental(self, veris_df):
        """
        Drops Environmental related rows and columns.
        
        Parameters
        ---------- 
        veris_df: DataFrame 
            Dataframe to be processed
        
        Returns
        ---------- 
        DataFrame
            The transformed DataFrame
        """

        veris_df_ = veris_df.copy()

        ## Drop rows
        veris_df_ = veris_df_.drop(veris_df_[veris_df_["action.Environmental"]].index) \
            .reset_index(drop=True)
        
        ## Drop cols
        veris_df_ = veris_df_.drop(columns=veris_df_.filter(like="environmental").columns)
        
        return veris_df_

    def get_task(self, task, veris_df):
        
        """ Defines the names of the predictor and target columns of veris_df dataset
            according to a selected task. Args:
        
        Parameters
        ---------- 
            task: str
                Opted task for learning
            
            veris_df: DataFrame 
                The veris_df dataset
        
        Returns
        ---------- 
        list
            a list of strings that contain the predictor columns
        list
            a list of strings that contain the target columns
        """
        veris_df_ = veris_df.copy()

        if task == 'attribute':
        
            c = veris_df_.filter(items=['attribute.Confidentiality']).columns
        
            i = veris_df_.filter(items=['attribute.Integrity']).columns
        
            a = veris_df_.filter(items=['attribute.Availability']).columns
        
            cia = list(itertools.chain(c, i, a))
        
            targets = cia
        
            predictors = ['action', 
                        'action.x.variety', 
                        'victim.industry', 
                        'victim.orgsize',
                        'asset.variety',
                        'asset.assets.variety'
                        ]
            return predictors, targets

        targets = veris_df_.filter(like=task).columns.tolist()
        
        targets = [x for x in targets if \
                "Other" not in x and 'Unknown' not in x]
        
        if task == "asset.variety":
        
            predictors = ['action', 
                        'action.x.variety', 
                        'victim.industry', 
                        'victim.orgsize']
        
            targets.remove("asset.variety.Embedded")
        
            targets.remove("asset.variety.Network")
        
        elif task.startswith("asset.assets.variety."):
        
            predictors = ['action', 
                        'action.x.variety', 
                        'victim.industry', 
                        'victim.orgsize',
                        'asset.variety']
        
        elif task == 'action':
        
            predictors = ['asset.variety', 
                        'asset.assets.variety', 
                        'victim.industry', 
                        'victim.orgsize']
        
            targets = ["action.Hacking", 
                    "action.Malware", 
                    "action.Error", 
                    "action.Misuse",
                    "action.Physical", 
                    "action.Social"]
        
        elif task.startswith("action") and task.endswith("variety"):
        
            predictors = ['asset.variety', 
                        'asset.assets.variety', 
                        'action',  
                        'victim.industry', 
                        'victim.orgsize',
                        'asset.variety']
        
        return predictors, targets

    def build_X(self, veris_df, predictors):\

        veris_df_ = veris_df.copy()

        X = pd.DataFrame(index=veris_df_.index)
        
        # predictors only for action prediction
        if 'asset.variety' in predictors:
            
            features = ['asset.variety.Embedded',
                        'asset.variety.Kiosk/Term',
                        'asset.variety.Media',
                        'asset.variety.Network',
                        'asset.variety.Person',
                        'asset.variety.Server',
                        'asset.variety.User Dev']
            
            X = pd.concat([X, veris_df_[features]],
                          join="inner", 
                          axis=1)
        
        if 'asset.assets.variety' in predictors:
            
            features = veris_df_ \
                .filter(regex='^asset.assets.variety.') \
                .columns \
                .tolist()
            
            # exclude Unknowns
            features = [f for f in features if 'Unknown' not in f]

            X = pd.concat([X, veris_df_[features]],
                                join='inner', 
                                axis=1)
        
        if 'action' in predictors:

            features = ['action.Error',
                        'action.Hacking', 
                        'action.Malware',
                        'action.Misuse', 
                        'action.Physical', 
                        'action.Social']
            
            X = pd.concat([X, veris_df_[features]],
                                join='inner', 
                                axis=1)

        if 'action.x.variety' in predictors:
            
            features = veris_df_.filter(regex='^(action\..*\.variety.)').columns.tolist()
            
            features = [f for f in features if 'Unknown' not in f]

            X = pd.concat([X, veris_df_[features]],
                                join='inner', 
                                axis=1)

        # standard predictors
        
        if 'victim.orgsize' in predictors:

            X = pd.concat([X, veris_df_['victim.orgsize.Large']],
                                join='inner', 
                                axis=1)

        if 'victim.industry' in predictors:
            
            X = pd.concat([X, veris_df_[['victim.industry.name']]],
                                join='inner', 
                                axis=1)
        
        return X

    def fetch_binary_target(self, veris_df, target_name):
        
        veris_df_ = veris_df.copy()
        
        y = veris_df_[target_name]
        
        # locate rows to drop
        
        dirty_rows_1 = (y != 0).values

        dirty_rows_2 = (y != 1).values
        
        dirty_rows = np.bitwise_and(dirty_rows_1, dirty_rows_2)

        # delete rows where the target's father is not activated (hierarchical classification model)
        irrelevant_rows = np.zeros(len(y)).astype(bool)

        if 'asset.assets.variety.' in target_name:
        
            mapping = {'asset.assets.variety.S ': 'asset.variety.Server',
                       'asset.assets.variety.T ': 'asset.variety.Kiosk/Term',
                       'asset.assets.variety.U ': 'asset.variety.User Dev',
                       'asset.assets.variety.M ': 'asset.variety.Media',
                       'asset.assets.variety.P ': 'asset.variety.Person'}
        
            father_col = mapping[target_name.split('-')[0]]
        
            irrelevant_rows = (veris_df_[father_col] == 0).values

        elif re.match(r'action.*\.variety', target_name):
        
            father_col = f'action.{target_name.split(".")[1].capitalize()}'
        
            irrelevant_rows = (veris_df_[father_col] == 0).values
        
        rows_to_drop = np.bitwise_or(irrelevant_rows, dirty_rows)
        
        rows_to_drop = np.where(rows_to_drop)[0].reshape(-1).tolist()

        # Drop columns from all datasets
        y = y.drop(rows_to_drop)

        veris_df_ = veris_df_.drop(rows_to_drop)

        return y.astype(int), veris_df_

    def merge_bruteforce_ddos_C2(self, veris_df):
        """
        Merges DoS, SQLi & Backdoor/C2 columns for action.hacking and action.malware
        keeping the maximum value amongst them
        
        Parameters
        ---------- 
        veris_df: DataFrame 
            Dataframe to be processed
        
        Returns
        ---------- 
        DataFrame
            The transformed DataFrame
        """
        veris_df_ = veris_df.copy()

        veris_df_['action.variety.Brute force'] = \
            veris_df_.loc[:, ['action.hacking.variety.Brute force', 
                              'action.malware.variety.Brute force'
                             ]
                         ] \
                     .max(axis=1)
        veris_df_ = veris_df_.drop(['action.malware.variety.Brute force'], axis=1)

        veris_df_['action.variety.DoS'] = \
            veris_df_.loc[:, ['action.hacking.variety.DoS', 
                              'action.malware.variety.DoS'
                             ]
                         ] \
                     .max(axis=1)   
        veris_df_ = veris_df_.drop(['action.malware.variety.DoS'], axis=1)

        veris_df_['action.hacking.variety.SQLi'] = \
            veris_df_.loc[:, ['action.hacking.variety.SQLi', 
                              'action.malware.variety.SQL injection'
                             ]
                         ] \
                     .max(axis=1)       
        veris_df_ = veris_df_.drop(['action.malware.variety.SQL injection'], axis=1)

        veris_df_['action.hacking.variety.Use of backdoor or C2'] = \
            veris_df_.loc[:, ['action.hacking.variety.Use of backdoor or C2', 
                              'action.malware.variety.Backdoor',
                              'action.malware.variety.C2'
                             ]
                         ] \
                     .max(axis=1)   

        veris_df_ = veris_df_.drop(['action.malware.variety.Backdoor'], axis=1)
        veris_df_ = veris_df_.drop(['action.malware.variety.C2'], axis=1)

        return veris_df_

def etl(veris_df, 
        task, 
        target,
        merge):

    logging.info("ETL\n")
    
    # Initiate ETL class
    pp = etl_maker()
    
    # Manage task predictors and features
    predictors, targets = pp.get_task(task, veris_df)

    # Let user choose specific target if not already done it from cli
    if target == ' ':

        pp = pprint.PrettyPrinter(indent=4)
    
        choices = dict(zip(range(len(targets)), targets))
    
        printer = pp.pprint(choices)
    
        target = choices[int(input(f"Select a target from the list above...\n{printer}\n"))]
    
    else:
    # Form target name
        target = " - ".join([task, target]) if task.startswith('asset.assets.variety') \
                                            else ".".join([task, target])

    # Drop environmental
    veris_df = pp.drop_environmental(veris_df)
    
    # Merge SQLi, Brute force, DoS, Backdoor&C2
    veris_df = pp.merge_bruteforce_ddos_C2(veris_df) if merge else veris_df

    # Output feature (if child get only rows where father activated)
    y, veris_df = pp.fetch_binary_target(veris_df, target)

    ## Creation of predictor dataset / Feature Selection
    X = pp.build_X(veris_df, predictors)

    return X, y, predictors, target