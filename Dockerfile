FROM python:3.9-slim

RUN apt-get update

RUN apt-get install -y git

RUN echo "Cloning into VCDB..." && git clone --quiet https://github.com/vz-risk/VCDB.git

RUN apt-get install -y libgomp1 gcc lsof build-essential graphviz

COPY ./ /code

WORKDIR /code 

RUN pip install --upgrade pip && pip install -r requirements.txt

