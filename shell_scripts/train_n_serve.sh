#!/bin/bash

curdir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

echo $curdir

train_n_serve() {
    if [[ $1 == *"asset.assets.variety"* ]]
    then
        modelname="${1} - ${2}"
    else
        modelname="${1}.${2}"
    fi
    # to train model only if only train is provided as argument
    if [[ $5 != "--no-training" ]]
    then
        for al in $3
        do
            mlflow run --experiment-name "$modelname" --entry-point train $curdir/.. -P task="$1" -P target="$2" -P algo="$al" --no-conda
        done
    fi
    if [[ $6 != "--no-serving" ]]
    then
        # kill not useful, can't find file and in any case serving stop by itself
        # $curdir/kill/kill_$modelname.sh
        mlflow models serve -m "models:/$modelname/Production" --host 0.0.0.0 --port $4 --no-conda &
    fi
}
