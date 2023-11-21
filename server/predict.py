import json
import os
import pickle
import typing as t
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


def _load_model(filename: str) -> tuple[DictVectorizer, Ridge]:
    with open(filename, "rb") as f:
        return pickle.load(f)


DV, MODEL = _load_model(os.path.join(os.path.dirname(__file__), "model.bin"))


@dataclass
class CoffeeInput:
    roaster: str
    roast: t.Literal["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"]
    roaster_country: str
    region_of_origin: str
    price_per_100g: float
    flavours: list[str]


def predict(coffee: CoffeeInput) -> float:
    X = DV.transform([asdict(coffee)])
    rating = np.expm1(MODEL.predict(X))[0]
    return float(rating)


def lambda_handler(event, context):
    data = json.loads(event["body"])
    coffee = CoffeeInput(**data)
    result = predict(coffee)
    return {"rating": result}
