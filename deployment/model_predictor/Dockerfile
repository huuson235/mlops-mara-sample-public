FROM python:3.10-slim

COPY deployment/model_predictor/requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./src /sample_solution/src
WORKDIR /sample_solution
