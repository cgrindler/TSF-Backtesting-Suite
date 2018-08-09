FROM python:3.6-slim

ADD requirements_Docker.txt /
RUN pip install -r /requirements_Docker.txt 

WORKDIR /
COPY ./ /
ENV PYTHONWARNINGS="ignore"
#CMD python /TimeSeriesForecast.py --operate --config /var/don/work
