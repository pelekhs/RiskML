from verispy import VERIS
import argparse
import sys
import os

# First Run the json_to_collapsed.R to create the collapsed csv

JSON_DIR = '../VCDB/data/json/validated/'
CSV_DIR = "../csv/"
FILE_NAME = "veris_df.csv"

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
    parser.add_argument("-s",
                        "--save_as",
                        help="the name of the csv to be saved",
                        default=FILE_NAME)
    return parser.parse_args()

def create_veris_csv(json_dir = JSON_DIR,
                     csv_dir = CSV_DIR,
                     fname = FILE_NAME):
    if  json_dir == None or csv_dir == None:
        print("Need json and collapsed csv directories")
        exit()
    v = VERIS(json_dir=json_dir)
    veris_df = v.json_to_df(verbose=False)
    veris_df.to_csv(os.path.join(csv_dir, fname))
    return veris_df

if __name__ == "__main__":
    args = parse_arguments()
    if len(sys.argv) < 3:
        print("Using:\n \"{}\" as JSON directory\n \"{}\" as CSV directory\n \"{}\" as file name"
              .format(args.json_dir, args.csv_dir, args.save_as))
    create_veris_csv(args.json_dir, args.csv_dir, args.save_as)
    print("Dataset stored as: {}".format(args.save_as))
