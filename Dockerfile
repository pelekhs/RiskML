FROM python:3.9-slim

RUN apt-get update

RUN apt-get install -y git lsof

RUN echo "Cloning into VCDB..." && git clone --quiet https://github.com/vz-risk/VCDB.git

RUN apt-get install -y libgomp1

COPY ./ /code

WORKDIR /code

RUN pip install --upgrade pip && pip install -r requirements.txt

