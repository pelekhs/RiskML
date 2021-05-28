#!/bin/bash -x
echo "Cloning VCDB ..."
(git clone https://github.com/vz-risk/VCDB.git temp \
&& mv temp/.git ../VCDB/.git) \
|| echo "Database update failed, proceeding with stored VCDB"
rm -rf temp