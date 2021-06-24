#!/bin/bash
source shell_scripts/utils.sh
echo "Training attribute..."
train_n_serve "attribute" "Confidentiality" "$1" 5030 "$2"
train_n_serve "attribute" "Integrity" "$1" 5031 "$2"
train_n_serve "attribute" "Availability" "$1" 5032 "$2"

