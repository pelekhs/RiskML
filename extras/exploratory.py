import globals
from preprocessing import preprocessor, ColToOneHot, load_datasets

if __name__ == "__main__":
    # Load data
    df, veris_df = load_datasets()

    # Data preprocessing = Feature selection, Imputation, One Hots etc...
    ## Drop environmental and modify NAICS to 2 digits
    pp = preprocessor(df, veris_df)
    df, veris_df = pp.drop_environmental()
    df = pp.round_NAICS(digits=2)

    # Pipeline 2: Predict 1st level asset
    ## Statistics
    print("\nAction statistics concerning incidents that belong to more than one actions:\n")
    print("Social-Error:")
    print(veris_df[veris_df["action.Social"]==1][veris_df["action.Error"]==1].index.shape)
    print("Hacking-Error:")
    print(veris_df[veris_df["action.Hacking"]==1][veris_df["action.Error"]==1].index.shape)
    print("Hacking-Malware:")
    print(veris_df[veris_df["action.Malware"]==1][veris_df["action.Hacking"]==1].index.shape)
    print("Error-Malware:")
    print(veris_df[veris_df["action.Malware"]==1][veris_df["action.Error"]==1].index.shape)
    print("Social-Malware:")
    print(veris_df[veris_df["action.Malware"]==1][veris_df["action.Social"]==1].index.shape)
    print("Social-Hacking:")
    print(veris_df[veris_df["action.Hacking "]==1][veris_df["action.Social"]==1].index.shape)