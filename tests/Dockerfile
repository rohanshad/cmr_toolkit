FROM ubuntu:24.04

RUN apt-get update && \
 apt-get upgrade -y && \
 apt-get install -y software-properties-common && \
 add-apt-repository ppa:deadsnakes/ppa
 
RUN apt-get install -y wget git && \
 apt-get install -y libsm6 libgl1 libxext6 libxrender-dev python3-pip python3.13 python3.13-venv

RUN python3.13 -m venv /venv

ENV PATH="/venv/bin:$PATH"

# Pin pip (hash-verified wheel)
RUN python -m pip install --no-cache-dir \
    "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    pip --version

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt