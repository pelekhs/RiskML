from verispy import VERIS

# Run the json_to_collapsed.R to create the collapsed csv

def create_veris_csv(json_dir = None,
                    csv_dir = None):
    if  json_dir == None or csv_dir == None:
        print("Need json and collapsed csv directories")
        exit()
    v = VERIS(json_dir=json_dir)
    veris_df = v.json_to_df(verbose=False)
    veris_df.to_csv(csv_dir)
    return veris_df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("NEED AS INPUT: 1. json dir \n2. dir to save csv \n Exiting...")
        exit()
    create_veris_csv(sys.argv[1], sys.argv[2])