from sklearn.preprocessing import LabelBinarizer
import pandas as pd

def preprocessing (X, y):

    # OneHotEncoding
    if 'victim.industry.name' in X.columns:
        lb = LabelBinarizer()
        transformed_column = lb.fit_transform(X['victim.industry.name'])
        industry_one_hot = \
            pd.DataFrame(data=transformed_column,
                         columns=['victim.industry.name.' + no for no in lb.classes_],
                         index=X.index)
        
        # drop dummy variable
        industry_one_hot = industry_one_hot.drop('victim.industry.name.Unknown', axis=1)

        X = X.drop('victim.industry.name', axis=1)
        
        X = pd.concat([X, industry_one_hot], axis=1)

    return X, y