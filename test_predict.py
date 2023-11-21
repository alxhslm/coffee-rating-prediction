import requests

url = "https://f727bsqbr5pihm444pfa7p4zf40jzfah.lambda-url.eu-west-2.on.aws/"

coffee = {
    "roaster": "Square Mile Coffee Roasters",
    "roast": "Light",
    "roaster_country": "England",
    "region_of_origin": "South America",
    "price_per_100g": 6.82,
    "flavours": ["fruity", "caramelly"],
}
response = requests.post(url, json=coffee)
response.raise_for_status()
print(response.json())
