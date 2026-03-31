FROM ubuntu:24.04

RUN apt-get update && \
 apt-get upgrade -y && \
 apt-get install -y software-properties-common && \
 add-apt-repository ppa:deadsnakes/ppa
 
RUN apt-get install -y wget git && \
 apt-get install -y libsm6 libgl1 libxext6 libxrender-dev python3-pip python3.13 python3.13-venv

RUN python3.13 -m venv /venv 

ENV PATH="/venv/bin:$PATH"

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt