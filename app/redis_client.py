import logging
import os

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    logging.info(f"Redis подключен: {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError:
    logging.warning("Redis недоступен, кэширование отключено")
    redis_client = None
