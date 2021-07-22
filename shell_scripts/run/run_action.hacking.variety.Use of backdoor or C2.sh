#!/bin/bash
curdir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source "${curdir}"/../train_n_serve.sh
train_n_serve "action.hacking.variety" "Use of backdoor or C2" "LGBM KNN RF" 5026
