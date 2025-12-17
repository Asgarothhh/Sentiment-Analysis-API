import json
import logging

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from requests.exceptions import Timeout, RequestException

from ml.model import load_model
from .redis_client import redis_client
from .utils import make_key

logging.basicConfig(level=logging.INFO,
                    filename="app.log",
                    filemode='w',
                    format="%(asctime)s-%(levelname)s-%(message)s")

app = FastAPI(title="Sentiment Analysis API", description="Определение тональности текста")
model = None


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float
    from_cache: bool


@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
        logging.info("Модель успешно загружена")
    except Exception as e:
        logging.error("Ошибка при вызове модели", exc_info=True)
        raise RuntimeError("Не удалось загрузить модель") from e


@app.post("/predict", response_model=SentimentResponse, summary="Анализ тональности")
def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым! Попробуйте снова")

    key = make_key(text)
    try:
        cached = redis_client.get(key)
    except redis.connection.ConnectionError:
        logging.warning("Redis недоступен. Сервис продолжате работу без кэша.")
        cached = None

    if cached:
        sentiment = json.loads(cached)
        return SentimentResponse(
            text=text,
            sentiment_label=sentiment["label"],
            sentiment_score=sentiment["score"],
            from_cache=True
        )

    logging.info(f"Получен текст: {text}")

    try:
        sentiment = model(text)
    except Timeout:
        logging.error("Таймаут при обращении к модели")
        raise HTTPException(status_code=504, detail="Сервис временно недоступен, попробуйте позже")
    except RequestException as e:
        logging.error(f"Ошибка при обращении к модели: {e}")
        raise HTTPException(status_code=502, detail="Ошибка при обработке текста, попробуйте снова")
    except Exception as e:
        logging.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    if not sentiment or not sentiment.label:
        raise HTTPException(status_code=500, detail="Модель не вернула результат, попробуйте снова")

    try:
        redis_client.setex(key, 86400, json.dumps({
            "text": text,
            "label": sentiment.label,
            "score": sentiment.score
        }))
    except redis.connection.ConnectionError:
        logging.warning("Redis недоступен. Результат не записан в кэш")

    return SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
        from_cache=False
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Глобальная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Произошла внутренняя ошибка сервера. Попробуйте позже."}
    )
