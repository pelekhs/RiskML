FROM python:3.9-slim-buster

# Install python packages
RUN set -xe && \
    apt-get update 

RUN apt-get install -y apt-utils

RUN apt-get install -y libpq-dev

RUN apt-get install -y build-essential

RUN pip install --upgrade pip

RUN pip install psycopg2 

RUN pip install mlflow==1.16.0 

RUN pip install boto3 
    