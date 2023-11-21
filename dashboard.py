import json

import requests
import streamlit as st

ROAST_STYLES = ["Light", "Medium-Light", "Medium", "Medium-Dark", "Dark"]

with open("data/roasters.json", "r") as f:
    ROASTERS: dict[str, str] = json.load(f)["known_roasters"]


with open("data/regions.json", "r") as f:
    REGIONS: list[str] = list(json.load(f).keys())


with open("data/flavours.json", "r") as f:
    FLAVOURS: list[str] = list(json.load(f).keys())


def _get_default_country(roaster: str) -> str:
    country = ROASTERS.get(roaster)
    if country is None:
        return "Other"
    return country


st.title(":coffee: Coffee rating predictor")

roaster: str = st.selectbox("Roaster", options=list(ROASTERS.keys()) + ["Other"])  # type:ignore
known_countries = sorted({v for v in ROASTERS.values()}) + ["Other"]
roaster_country = st.selectbox(
    "Roaster country", options=known_countries, index=known_countries.index(_get_default_country(roaster))
)
roast = st.selectbox("Roast style", options=ROAST_STYLES)

origin = st.selectbox("Region of origin", options=REGIONS + ["Other"])

price_mode = st.radio("Select price", options=["Price per bag", "Price per 100g"], horizontal=True)
if price_mode == "Price per bag":
    col1, col2 = st.columns(2)
    with col1:
        price_per_bag = st.number_input("Enter price per bag in USD")
    with col2:
        bag_weight = st.number_input("Enter weight of bag in g", value=200)
    price_per_100g = 100 * price_per_bag / bag_weight
else:
    price_per_100g = st.number_input("Enter price per 100g in USD")


flavours = st.multiselect("Flavour profile", options=FLAVOURS, format_func=lambda s: s.capitalize())

url = "https://f727bsqbr5pihm444pfa7p4zf40jzfah.lambda-url.eu-west-2.on.aws/"

coffee = {
    "roaster": roaster,
    "roast": roast,
    "roaster_country": roaster_country,
    "region_of_origin": origin,
    "price_per_100g": price_per_100g,
    "flavours": flavours,
}
response = requests.post(url, json=coffee)
response.raise_for_status()
prediction = response.json()
st.metric("Predicted rating", value="{:.2f}%".format(prediction["rating"]))
