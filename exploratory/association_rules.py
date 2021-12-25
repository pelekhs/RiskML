import pandas as pd
import os
from dotenv import load_dotenv
from verispy import VERIS
from mlxtend.frequent_patterns import association_rules, fpgrowth

# Environment variables
load_dotenv() 

JSON_DIR = os.environ.get('JSON_DIR')

CSV_DIR = os.path.join("..", os.environ.get('CSV_DIR'))

VERIS_DF = os.environ.get('BOOLEAN_CSV_NAME')

VERIS_DF_URL = os.environ.get('BOOLEAN_CSV_URL')

enemies = pd.read_csv('bool.csv')
enemies.head()

# enemies = enemies[enemies['timeline.incident.year']>=2000]
# ext = [col for col in enemies.columns if 'actor.external.country.' in col]
# vict = [col for col in enemies.columns if 'victim.country.' in col]
# enemies =  enemies[ext+vict]

# Building the model 
frq_items = fpgrowth(enemies, min_support = 0.002, use_colnames = True) 
# Collecting the inferred rules inside the enemies dataframe 
rules_countries = association_rules(frq_items, metric ="lift",\
                                    min_threshold = 1) 
rules_countries = rules_countries.sort_values(['confidence', 'lift'],\
                                              ascending =[False, False])

## Mainly US <-> US

rules_countries[0:20].to_csv("vcdb_freqs.csv", index=False)