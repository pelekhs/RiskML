#!/bin/bash
curdir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source "${curdir}"/../train_n_serve.sh
echo "Training models..."
train_n_serve "asset.variety" "Server" "LGBM" 5010
train_n_serve "asset.variety" "User Dev" "LGBM" 5011
train_n_serve "asset.variety" "Network" "LGBM" 5012
train_n_serve "asset.assets.variety.S" "Database" "LGBM" 5013
train_n_serve "asset.assets.variety.S" "Web application" "LGBM" 5014
train_n_serve "asset.assets.variety.U" "Desktop or laptop" "LGBM" 5015
train_n_serve "action" "Malware" "LGBM" 5020
train_n_serve "action" "Hacking" "LGBM" 5021
train_n_serve "action.malware.variety" "Ransomware" "LGBM" 5022
train_n_serve "action.hacking.variety" "DoS" "LGBM" 5023
train_n_serve "action.hacking.variety" "SQLi" "LGBM" 5024
train_n_serve "action.hacking.variety" "Brute force" "LGBM" 5025
train_n_serve "action.hacking.variety" "Use of backdoor or C2" "LGBM" 5026
train_n_serve "attribute" "Confidentiality" "LGBM" 5030
train_n_serve "attribute" "Integrity" "LGBM" 5031
train_n_serve "attribute" "Availability" "LGBM" 5032
tail -F "KeepingContainerAlive"