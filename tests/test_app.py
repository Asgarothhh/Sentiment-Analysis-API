import pytest
import requests


@pytest.mark.parametrize(
    "input_text, expected_label",
    [
        ("Этот сервис ужасен, никогда больше не вернусь", "negative"),
        ("Мне очень понравилось обслуживание, всё было на высшем уровне", "positive"),
        ("Сегодня среда, завтра четверг", "neutral"),
    ],
)
def test_sentiment(input_text: str, expected_label: str):
    response = requests.post(
        "http://127.0.0.1:8080/predict",
        json={"text": input_text},
    )
    data = response.json()
    assert data["text"] == input_text.strip()
    assert data["sentiment_label"] == expected_label
    assert isinstance(data["sentiment_score"], float)
    assert "from_cache" in data
