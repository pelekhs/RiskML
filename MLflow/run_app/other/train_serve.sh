#!/bin/bash

# train
python ./run_app/run_asset_variety.py &&
(mlflow models serve -m "models:/asset.variety.Server/Production" --host 0.0.0.0 --port 5010 --no-conda &
mlflow models serve -m "models:/asset.variety.User Dev/Production" --host 0.0.0.0 --port 5011 --no-conda &
mlflow models serve -m "models:/asset.variety.Network/Production" --host 0.0.0.0 --port 5012 --no-conda &)


# serve
#for experiment in Server User Dev Network
#do
#	
#done



tail -F anything
