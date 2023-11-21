import requests

hostname = "coffee_server"
url = f"http://{hostname}:8501/predict"
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
