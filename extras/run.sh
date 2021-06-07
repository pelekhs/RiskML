#!/bin/bash
source run_app/utils.sh
echo "Training assets..."
train_n_serve "attribute" "Confidentiality" "$1" 5030
train_n_serve "attribute" "Integrity" "$1" 5031
train_n_serve "attribute" "Availability" "$1" 5032
train_n_serve "asset.variety" "Server" "$1" 5010
train_n_serve "asset.variety" "User Dev" "$1" 5011
train_n_serve "asset.variety" "Network" "$1" 5012
train_n_serve "asset.assets.variety.S" "Database" "$1" 5013
train_n_serve "asset.assets.variety.S" "Web application" "$1" 5014
train_n_serve "asset.assets.variety.U" "Desktop or laptop" "$1" 5015
train_n_serve "action" "Malware" "$1" 5020
train_n_serve "action" "Hacking" "$1" 5021
train_n_serve "action.malware.variety" "Ransomware" "$1" 5022
train_n_serve "action.hacking.variety" "DoS" "$1" 5023
train_n_serve "action.hacking.variety" "SQLi" "$1" 5024
train_n_serve "action.hacking.variety" "Brute force" "$1" 5025
train_n_serve "action.hacking.variety" "Use of backdoor or C2" "$1" 5026
