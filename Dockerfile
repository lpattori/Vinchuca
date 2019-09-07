FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt
COPY server.py server.py
COPY app app/

RUN python server.py load

EXPOSE $PORT
CMD python server.py serve ${PORT:-5000}
