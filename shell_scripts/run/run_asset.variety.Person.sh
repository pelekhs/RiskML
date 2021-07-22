#!/bin/bash
curdir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source "${curdir}"/../train_n_serve.sh
train_n_serve "asset.variety" "Person" "LGBM RF KNN" 5009
