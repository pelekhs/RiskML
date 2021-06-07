#!/bin/bash -x
echo "Cloning VCDB ..."
(git clone https://github.com/vz-risk/VCDB.git temp \
&& rm -rf ../VCDB/.git && mv temp/.git ../VCDB/) \
|| echo "Database update failed, proceeding with stored VCDB"
rm -rf temp