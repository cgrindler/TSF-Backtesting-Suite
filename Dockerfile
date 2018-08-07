FROM python:3.6-slim

ADD requirements.txt /
RUN pip install -r /requirements.txt 
WORKDIR /
COPY ./ /
RUN python /TimeSeriesForecast.py
