#!/bin/bash -x
echo "Proceeding to training..."
./shell_scripts/train_attribute.sh "KNN RF LGBM" "--no-training"
./shell_scripts/train_asset.sh "LGBM RF KNN" "--no-training"
./shell_scripts/train_action.sh "LGBM RF KNN" "--no-training"
tail -F "KeepingContainerAlive"