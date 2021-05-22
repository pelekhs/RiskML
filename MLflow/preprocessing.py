from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import logging

# Configure logging
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

class OneHotTransformer(BaseEstimator, TransformerMixin):

  
  # add another additional parameter, just for fun, while we are at it
  def __init__(self, feature_name, feature_labels):  
    
    self.feature_name = feature_name
    self.feature_labels = feature_labels
    self.lb = LabelBinarizer()

  def fit(self, X, y=None):

    self.lb.fit(self.feature_labels)

    return self
    

  def transform(self, X, y=None):
    
    X_ = X.copy()
    
    transformed_column = self.lb.transform(X_[self.feature_name]) 

    industry_one_hot = \
        pd.DataFrame(data=transformed_column,
                     columns=[f'{self.feature_name}.' + no for no in self.feature_labels],
                     index=X_.index
                    )
    
    # drop dummy variable
    industry_one_hot = industry_one_hot.drop(f'{self.feature_name}.Unknown', 
                                             axis=1)

    X_ = X_.drop(self.feature_name, axis=1)
    
    X_ = pd.concat([X_, industry_one_hot], 
                   join='inner',
                   axis=1)

    return X_.astype(float)

class myPCA(BaseEstimator, TransformerMixin):
  
  # add another additional parameter, just for fun, while we are at it
  def __init__(self, n_components):  
    
    self.n_components = n_components
    self.pca = PCA(n_components=n_components)
    
  def fit(self, X, y=None):
    
    if self.n_components != 0:

      self.pca.fit(X)
      logging.info(f'\nExplained variance ratio: {self.pca.explained_variance_ratio_}')

    return self
    
  def transform(self, X, y=None):

    if self.n_components != 0:
      
      X_ = X.copy()
      return self.pca.transform(X_)
    
    else:

      return X