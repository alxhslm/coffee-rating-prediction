import os

import requests
from aws_requests_auth.aws_auth import AWSRequestsAuth

resource = "f727bsqbr5pihm444pfa7p4zf40jzfah"
region = "eu-west-2"
host = f"{resource}.lambda-url.{region}.on.aws"
url = f"https://{host}/"

coffee = {
    "roaster": "Square Mile Coffee Roasters",
    "roast": "Light",
    "roaster_country": "England",
    "region_of_origin": "South America",
    "price_per_100g": 6.82,
    "flavours": ["fruity", "caramelly"],
}

auth = AWSRequestsAuth(
    aws_access_key=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_host=host,
    aws_region=region,
    aws_service="lambda",
)
response = requests.post(url, json=coffee, auth=auth)
response.raise_for_status()
print(response.json())
