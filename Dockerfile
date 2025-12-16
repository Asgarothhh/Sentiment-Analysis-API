FROM python:3.12

RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py /workdir/
COPY app/ /workdir/app/
COPY ml/ /workdir/ml/

WORKDIR /workdir

RUN pip install -U -e .

CMD ["sh", "-c", "redis-server --daemonize yes && uvicorn app.app:app --host 127.0.0.1 --port 80"]