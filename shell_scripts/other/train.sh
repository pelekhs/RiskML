#!/bin/bash

# asset

mlflow run --experiment-name 'asset.variety' --entry-point train . -P task='asset.variety' -P target='Server' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'asset.variety' --entry-point train . -P task='asset.variety' -P target='User Dev' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'asset.variety' --entry-point train . -P task='asset.variety' -P target='Network' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \

mlflow run --experiment-name 'asset.assets.variety' --entry-point train . -P task='asset.assets.variety.S' -P target='Database' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'asset.assets.variety' --entry-point train . -P task='asset.assets.variety.S' -P target='Web application' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'asset.assets.variety' --entry-point train . -P task='asset.assets.variety.U' -P target='Desktop or laptop' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \

# action


mlflow run --experiment-name 'action' --entry-point train . -P task='action' -P target='Malware' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'action' --entry-point train . -P task='action' -P target='Hacking' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \

mlflow run --experiment-name 'action.malware.variety' --entry-point train . -P task='action.malware.variety' -P target='Ransomware' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'action.hacking.variety' --entry-point train . -P task='action.hacking.variety' -P target='DoS' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'action.hacking.variety' --entry-point train . -P task='action.hacking.variety' -P target='SQLi' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'action.hacking.variety' --entry-point train . -P task='action.hacking.variety' -P target='Brute force' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda && \
mlflow run --experiment-name 'action.hacking.variety' --entry-point train . -P task='action.hacking.variety' -P target='Use of backdoor or C2' -P train_size=1 -P n_folds=5 -P algo='LGBM' -P split_random_state=0 -P merge=1 -P pca=0 --no-conda
