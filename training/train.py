#!/usr/bin/env python
# coding: utf-8

import json
import os
import pickle
import re
import typing as t

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# parameters

alpha = 1.0
n_splits = 5
output_file = "model.bin"

# data preparation
with open("data/roasters.json", "r") as f:
    ROASTERS: dict[str, t.Any] = json.load(f)

with open("data/regions.json", "r") as f:
    REGIONS: dict[str, list[str]] = json.load(f)

with open("data/flavours.json", "r") as f:
    FLAVOURS: dict[str, list[str]] = json.load(f)

df = pd.read_csv(os.path.join("data/simplified_coffee.csv"))
for col in ["name", "roaster", "roast", "loc_country", "origin", "review"]:
    df[col] = df[col].astype("string")

df["review_date"] = pd.to_datetime(df["review_date"])
df = df.rename(columns={"loc_country": "roaster_country", "100g_USD": "price_per_100g", "origin": "country_of_origin"})
df["roast"] = df["roast"].fillna(df["roast"].mode().iloc[0])
df["roaster_country"] = df["roaster_country"].str.replace("New Taiwan", "Taiwan")

replace = {"’s": "'s", "é": "e", "’": "'"}  # noqa: RUF001
for k, v in replace.items():
    df["roaster"] = df["roaster"].str.replace(k, v)


def _invert_region_map(regions: dict[str, list[str]]) -> dict[str, str]:
    map = {}
    for r, countries in regions.items():
        for c in countries:
            map[c] = r
    return map


df["region_of_origin"] = df["country_of_origin"].map(_invert_region_map(REGIONS)).fillna("Other")
df["roaster"] = df["roaster"].where(df["roaster"].apply(lambda r: r in ROASTERS["popular_roasters"]), "Other")


def rating_contains_words(review: str, keywords: list[str]) -> bool:
    def _extract_words(string: str) -> list[str]:
        return re.findall(r'\w+', string.lower())

    words = _extract_words(review)
    for w in keywords:
        if w in words:
            return True
    return False


for flavour, keywords in FLAVOURS.items():
    df[flavour] = df["review"].apply(rating_contains_words, args=(keywords,))
df["flavours"] = df.apply(lambda coffee: [flavour for flavour in FLAVOURS if coffee[flavour]], axis=1)


df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=1)

features = ["roaster", "roast", "roaster_country", "region_of_origin", "price_per_100g", "flavours"]
target = "rating"


# training
def train(X: pd.DataFrame, y: pd.Series, alpha: float) -> tuple[DictVectorizer, Ridge]:
    dicts = X.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    y = y.apply(np.log1p)

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    return dv, model


def predict(df: pd.DataFrame, dv: DictVectorizer, model: Ridge) -> pd.Series:
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return pd.Series(np.expm1(y_pred), index=df.index, dtype=float, name=target)


# validation
print(f'doing validation with alpha={alpha}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = pd.Series(name="rmse", dtype=float)
for fold, (train_idx, val_idx) in enumerate(kfold.split(df_train_val)):
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    dv, model = train(df_train[features], df_train[target], alpha=alpha)
    y_pred = predict(df_val[features], dv, model)

    scores.loc[fold] = mean_squared_error(df_val[target], y_pred, squared=False)

    print(f'rmse on fold {fold} is {scores[fold]}')


print('validation results:')
print('alpha=%.3g %.3f +- %.3f' % (alpha, scores.mean(), scores.std()))


# training the final model
print('training the final model with alpha={alpha}')

dv, model = train(df_train_val[features], df_train_val[target], alpha=alpha)
y_pred = predict(df_test[features], dv, model)
rmse = mean_squared_error(df_test[target], y_pred, squared=False)

print(f'rmse={rmse}')


# Save the model
print(f'saving model to {output_file}')
with open(os.path.join("server", output_file), 'wb') as f_out:
    pickle.dump((dv, model), f_out)


print('training complete!')
