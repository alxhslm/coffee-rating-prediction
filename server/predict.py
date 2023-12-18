import json
import os
import pickle
import typing as t
from dataclasses import asdict, dataclass

import numpy as np
import scipy.stats as sps
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


def _load_model(filename: str) -> tuple[DictVectorizer, Ridge, dict[str, float]]:
    with open(filename, "rb") as f:
        return pickle.load(f)


DV, MODEL, DATA = _load_model(os.path.join(os.path.dirname(__file__), "model.bin"))


@dataclass
class CoffeeInput:
    roaster: str
    roast: t.Literal["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"]
    roaster_country: str
    country_of_origin: str
    price_per_100g: float
    flavours: list[str]


def _preprocess_features(coffee: CoffeeInput) -> CoffeeInput:
    return CoffeeInput(**(asdict(coffee) | {"price_per_100g": np.log1p(coffee.price_per_100g)}))


def predict(coffee: CoffeeInput) -> float:
    coffee = _preprocess_features(coffee)
    X = DV.transform([asdict(coffee)])
    rating = MODEL.predict(X)[0]
    return float(rating)


def lambda_handler(event, context):
    data = json.loads(event["body"])
    coffee = CoffeeInput(**data)
    result = predict(coffee)
    dist = sps.norm(loc=DATA["mean"], scale=DATA["std"])
    return {"rating": result, "percentile": dist.cdf(result) * 100}
