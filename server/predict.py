import os
import pickle
import typing as t

import numpy as np
from flask import Flask, Response, jsonify, request
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


def _load_model(filename: str) -> tuple[DictVectorizer, Ridge]:
    with open(filename, "rb") as f:
        return pickle.load(f)


DV, MODEL = _load_model(os.path.join(os.path.dirname(__file__), "model.bin"))


def _predict(coffee: dict[str, t.Any]) -> float:
    X = DV.transform([coffee])
    rating = np.expm1(MODEL.predict(X))[0]
    return float(rating)


app = Flask("app")


@app.route('/predict', methods=['POST'])
def predict() -> Response:
    coffee = request.get_json()
    result = {'rating': _predict(coffee)}
    return jsonify(result)  ## send back the data in json format to the user


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8501")
