#!/bin/bash
source run_app/train_n_serve_utils.sh
echo "Training action..."
train_n_serve "action" "Malware" "$1" 5020
train_n_serve "action" "Hacking" "$1" 5021
train_n_serve "action.malware.variety" "Ransomware" "$1" 5022
train_n_serve "action.hacking.variety" "DoS" "$1" 5023
train_n_serve "action.hacking.variety" "SQLi" "$1" 5024
train_n_serve "action.hacking.variety" "Brute force" "$1" 5025
train_n_serve "action.hacking.variety" "Use of backdoor or C2" "$1" 5026
#train_n_serve "action.hacking.variety" "MitM" "${algos}" 5027
