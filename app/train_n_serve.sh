#!/bin/bash -x
echo "Proceeding to training..."
./run_app/train_attribute.sh "KNN RF LGBM"
./run_app/train_asset.sh "LGBM RF KNN"
./run_app/train_action.sh "LGBM RF KNN" 
tail -F "KeepingContainerAlive"