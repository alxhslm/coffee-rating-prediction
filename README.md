# :coffee: Coffee rating prediction
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://coffee-rating-prediction.streamlit.app/)

The objective of this project is to be able to predict how highly rated a coffee would be on [Coffee Review](https://www.coffeereview.com/), based upon some meta-data about the selected coffee.

## Downloading the dataset
The dataset can be download from [Kaggle](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset/data). You can either download this:
- Directly from the website
- Using the Kaggle API as follows:
    ``` bash
    !kaggle datasets download -d schmoyote/coffee-reviews-dataset -f simplified_coffee.csv -p training
    ```
## Training the model
To train a linear regression model, run the following script:

``` bash
training/train.py
```

## Generating predictions using the model
The model is served within a separate Docker container. This is hosted using [AWS Lambda](https://aws.amazon.com/lambda/).

To build the server locally, run the following command:
``` bash
docker build server -t coffee-prediction-server:latest
```

To start the server for the first time, use the following command:
``` bash
docker run -it --network coffee-rating-prediction_devcontainer_default --hostname coffee_server --name coffee-prediction-server coffee-prediction-server:latest
```

To restart the server, run the following:
``` bash
docker start -i coffee-prediction-server
```

Then run the following to test the server:
``` bash
python test_predict.py
```

## Using the interactive dashboard
To investigate the predicted rating of a given coffee, you can use the interactive `streamlit` dashboard. This is hosted on [Streamlit Cloud](https://streamlit.io/cloud).

You can also launch the dashboard locally by running the following command:

``` bash
streamlit run dashboard.py
```
