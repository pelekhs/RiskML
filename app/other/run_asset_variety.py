import mlflow, os, sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)
from train import train_evaluate_register

arguments = {
    'algos': ['LGBM'],
    'targets': ['Server', 'User Dev', 'Network']
    }

# Create an experiment name, which must be unique and case sensitive
for algo in arguments['algos']:
    for target in arguments['targets']:
        experiment_id = mlflow.create_experiment('asset.variety.' + target)
        train_evaluate_register(task='asset.variety', 
                                target=target, 
                                algo=algo, 
                                hyperparams=None, 
                                imputer=None, 
                                merge=1,
                                train_size=1, 
                                split_random_state=0, 
                                n_folds=5,
                                pca=0)
                                
        experiment = mlflow.get_experiment(experiment_id)
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))