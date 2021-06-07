#!/bin/bash
source shell_scripts/utils.sh
echo "Training assets..."
train_n_serve "asset.variety" "Server" "$1" 5010 "$2" 
train_n_serve "asset.variety" "User Dev" "$1" 5011  "$2"
train_n_serve "asset.variety" "Network" "$1" 5012 "$2"
train_n_serve "asset.assets.variety.S" "Database" "$1" 5013 "$2"
train_n_serve "asset.assets.variety.S" "Web application" "$1" 5014 "$2"
train_n_serve "asset.assets.variety.U" "Desktop or laptop" "$1" 5015 "$2"


