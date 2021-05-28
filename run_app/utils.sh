#!/bin/bash

train_n_serve() {
    if [[ $1 == *"asset.assets.variety"* ]]
    then
        modelname="${1} - ${2}"
    else
        modelname="${1}.${2}"
    fi
    for al in $3
    do
        mlflow run --experiment-name "$modelname" --entry-point train . -P task="$1" -P target="$2" -P train_size=1 -P n_folds=5 -P algo="$al" -P split_random_state=0 -P merge=1 --no-conda
    done
    mlflow models serve -m "models:/$modelname/Production" --host 0.0.0.0 --port $4 --no-conda &
}