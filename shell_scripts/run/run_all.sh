#!/bin/bash
curdir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source "${curdir}"/../train_n_serve.sh
echo "Training models..."
train_n_serve "asset.variety" "Server" "LGBM KNN RF" 5010
train_n_serve "asset.variety" "User Dev" "LGBM KNN RF" 5011
train_n_serve "asset.variety" "Network" "LGBM KNN RF" 5012
train_n_serve "asset.assets.variety.S" "Database" "LGBM KNN RF" 5013
train_n_serve "asset.assets.variety.S" "Web application" "LGBM KNN RF" 5014
train_n_serve "asset.assets.variety.U" "Desktop or laptop" "LGBM KNN RF" 5015
train_n_serve "action" "Malware" "LGBM KNN RF" 5020
train_n_serve "action" "Hacking" "LGBM KNN RF" 5021
train_n_serve "action.malware.variety" "Ransomware" "LGBM KNN RF" 5022
train_n_serve "action.hacking.variety" "DoS" "LGBM KNN RF" 5023
train_n_serve "action.hacking.variety" "SQLi" "LGBM KNN RF" 5024
train_n_serve "action.hacking.variety" "Brute force" "LGBM KNN RF" 5025
train_n_serve "action.hacking.variety" "Use of backdoor or C2" "LGBM KNN RF" 5026
train_n_serve "attribute" "Confidentiality" "LGBM KNN RF" 5030
train_n_serve "attribute" "Integrity" "LGBM KNN RF" 5031
train_n_serve "attribute" "Availability" "LGBM KNN RF" 5032
tail -F "KeepingContainerAlive"