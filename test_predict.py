import requests

hostname = "coffee_server"
url = f"http://{hostname}:8501/predict"
coffee = {
    "roaster": "Square Mile Coffee Roasters",
    "roast": "Light",
    "roaster_country": "England",
    "origin": "Colombia",
    "100g_USD": 6.82,
    "review": "Tasting note: BLACKCURRANT, PINEAPPLE, CANDY. Soooo goood! We're thrilled to welcome a first-time "
    "showcase, El Trapiche from Nariño, Colombia, to our shop. A tropical fruit bomb produced by Yhon David Gomez, "
    "if you're in the mood for something out of the ordinary - think a washed coffee tasting like a natural - El "
    "Trapiche has your name on it.",
}
prediction = requests.post(url, json=coffee).json()
print(prediction)
