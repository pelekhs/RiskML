import mlflow
import os
import click
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv
load_dotenv()


def print_run_info(runs):
    for r in runs:
        print("run_id: {}".format(r.info.run_id))
        # print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
        # print("metrics: {}".format(r.data.metrics))

        # Exclude mlflow system tags
        # tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        # print("tags: {}".format(tags))

@click.command()
@click.option("--experiments", "-e", 
              type=str,
              multiple=True,
              help="The experiment name for which you need to download the example file. (Repeat prefix for multiple arguments)"
              )
@click.option("--root-dir", "-d", 
              type=str,
              multiple=False,
              default='.',
              help="The local (root) directory to use for downloading the files"
              )
def run(experiments, root_dir):
    # Search all runs under experiment id and order them by
    # descending value of the metric 'm'
    # root_dir = "./download_examples"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    client = MlflowClient(tracking_uri='http://localhost:5000')
    for name in experiments:
        e = client.get_experiment_by_name(name)
        print(e.experiment_id, e.name)
        # define / create local directory to store example
        local_dir = os.path.join(root_dir, e.name)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        runs = client.search_runs(e.experiment_id)
        print_run_info(runs)
        print("--")
        # download last run artifacts
        local_path = client.download_artifacts(runs[0].info.run_id, "model/input_example.json", local_dir)
        print("Artifacts downloaded in: {}".format(local_dir))
        print("Artifacts: {}".format(local_dir))

if __name__ == '__main__':
    run()