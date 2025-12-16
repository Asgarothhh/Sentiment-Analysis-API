from pathlib import Path

import yaml
from pydantic import BaseModel
from transformers import pipeline

config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class SentimentPrediction(BaseModel):
    """Class representing a sentiment prediction result."""
    label: str
    score: float


def load_model():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """
    model_hf = pipeline(config['task'], model=config['model'], device=-1)

    def model(text: str) -> SentimentPrediction:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class['label'],
            score=pred_best_class['score']
        )

    return model
