#!/bin/bash
source run_app/train_n_serve_utils.sh
echo "Training attribute..."
train_n_serve "attribute" "Confidentiality" "$1" 5030 
train_n_serve "attribute" "Integrity" "$1" 5031
train_n_serve "attribute" "Availability" "$1" 5032

