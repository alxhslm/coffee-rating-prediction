#!/usr/bin/env python
# coding: utf-8

import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# parameters

alpha = 1.0
n_splits = 5
output_file = 'model.bin'
regions_file = "regions.json"
flavours_file = "flavours.json"

# data preparation

df = pd.read_csv(os.path.join("data/simplified_coffee.csv"))
for col in ["name", "roaster", "roast", "loc_country", "origin", "review"]:
    df[col] = df[col].astype("string")

df["review_date"] = pd.to_datetime(df["review_date"])
df = df.rename(columns={"loc_country": "roaster_country"})
df["roast"] = df["roast"].fillna(df["roast"].mode().iloc[0])
df["roaster_country"] = df["roaster_country"].str.replace("New Taiwan", "Taiwan")


def _get_region_map(filename: str) -> dict[str, str]:
    with open(os.path.join("data", filename), "r") as f:
        REGIONS = json.load(f)
    regions = {}
    for r, countries in REGIONS.items():
        for c in countries:
            regions[c] = r
    return regions


regions = _get_region_map(regions_file)
df["region"] = df["origin"].map(regions).fillna("Other")

roasters = df["roaster"].value_counts()
ROASTERS = roasters[roasters > 10].index
df["roaster"] = df["roaster"].where(df["roaster"].apply(lambda r: r in ROASTERS), "Other")


def _get_flavour_map(filename: str) -> dict[str, list[str]]:
    with open(os.path.join("data", filename), "r") as f:
        return json.load(f)  # type:ignore


def rating_contains_words(review: str, keywords: list[str]) -> bool:
    def _extract_words(string: str) -> list[str]:
        return re.findall(r'\w+', string.lower())

    words = _extract_words(review)
    for w in keywords:
        if w in words:
            return True
    return False


flavours = _get_flavour_map(flavours_file)
for flavour, keywords in flavours.items():
    df[flavour] = df["review"].apply(rating_contains_words, args=(keywords,))


df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=1)

features = ["roaster", "roast", "roaster_country", "region", "100g_USD"] + list(flavours.keys())
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
