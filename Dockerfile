#Dockerfile, Image, Container
FROM python:3.8.6-buster

# RUN pip install python-multipart
RUN pip install --upgrade pip
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt


COPY model /model
#COPY data_sources /data_sources
#COPY interface /interface
#COPY ml_logic /ml_logic
#COPY notebook /notebook
#COPY app.py /app.py
COPY api.py /api.py

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
