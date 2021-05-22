import mlflow

class ModelOut (mlflow.pyfunc.PythonModel):
     def __init__(self, model):
          self.model = model
    
     def predict (self, context, model_input):
          return self.model.predict_proba(model_input)[:,1]

mlflow_serve_conda_env ={'channels': ['defaults'],
                         'name':'conda',
                         'dependencies': [ 'python=3.9', 'pip',
                         {'pip': ['mlflow',
                                  'scikit-learn',
                                  'cloudpickle',
                                  'pandas',
                                  'numpy', 
                                  'lightgbm']}]}