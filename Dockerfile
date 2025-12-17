FROM python:3.12-slim

WORKDIR /workdir

COPY requirements.txt requirements-dev.txt setup.py ./
COPY app/ /workdir/app/
COPY ml/ /workdir/ml/

RUN pip install -U -e .

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
