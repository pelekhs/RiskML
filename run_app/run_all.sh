#!/bin/bash -x

echo "Cloning VCDB ..."
(git clone https://github.com/vz-risk/VCDB.git temp && mv temp/.git ../VCDB/.git) || echo "Clone failed, proceeding with stored VCDB"
rm -rf temp
echo "Proceeding to training"
./run_app/train_attribute.sh 'LGBM RF' && \
./run_app/train_asset.sh 'LGBM' && \
./run_app/train_action.sh 'LGBM'
tail -F anything