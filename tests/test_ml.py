import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from ml.model import SentimentPrediction, load_model


@pytest.fixture(scope="function")
def model():
    return load_model()


@pytest.mark.parametrize(
    "text",
    [
        "Этот сервис ужасен, никогда больше не вернусь",
        "Мне очень понравилось обслуживание, всё было на высшем уровне",
        "Сегодня среда, завтра четверг",
    ],
)
def test_smoke_sentiment(model, text: str):
    model_pred = model(text)
    assert isinstance(model_pred, SentimentPrediction)
    assert model_pred.label in {"positive", "negative", "neutral"}


def test_quality_sentiment(model):
    texts = [
        "Этот сервис ужасен, никогда больше не вернусь",
        "Фильм был скучным и затянутым",
        "Еда холодная и невкусная",
        "Мне очень понравилось обслуживание, всё было на высшем уровне",
        "Прекрасный день, настроение отличное!",
        "Результаты превзошли все ожидания",
        "Сегодня среда, завтра четверг",
        "Температура воздуха около двадцати градусов",
        "Я поеду в магазин, потом домой",
        "Фильм был интересный, но слишком длинный",
        "Обслуживание хорошее, но цены высокие",
        "Мне понравился сюжет, но актёры играли плохо",
    ]
    expected_labels = [
        "negative", "negative", "negative",
        "positive", "positive", "positive",
        "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral",
    ]

    preds = [model(t).label for t in texts]
    report = classification_report(expected_labels, preds, zero_division=0)

    print("\nКачественный отчёт по модели:\n", report)
    acc = accuracy_score(expected_labels, preds)
    assert acc > 0.7
