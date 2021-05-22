#!/bin/bash

echo "Serving models..."
mlflow models serve -m "models:/asset.variety.Server/Production" --port 5010 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/asset.variety.User Dev/Production" --port 5011 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/asset.variety.Network/Production" --port 5012 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/asset.assets.variety.S - Database/Production" --port 5013 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/asset.assets.variety.S - Web application/Production" --port 5014 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/asset.assets.variety.U - Desktop or laptop/Production" --port 5015 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.Hacking/Production" --port 5020 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.Malware/Production" --port 5021 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.malware.variety.Ransomware/Production" --port 5022 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.hacking.variety.DoS/Production" --port 5023 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.hacking.variety.SQLi/Production" --port 5024 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.hacking.variety.Brute force/Production" --port 5025 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/action.hacking.variety.Use of backdoor or C2/Production" --port 5026 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/attribute.Confidentiality/Production" --port 5030 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/attribute.Integrity/Production" --port 5031 --host 0.0.0.0 --no-conda &
mlflow models serve -m "models:/attribute.Availability/Production" --port 5032 --host 0.0.0.0 --no-conda