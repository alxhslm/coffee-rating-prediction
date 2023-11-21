# :coffee: Coffee rating prediction
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

## Predicting ratings using the model
Build the server using the following command:
```
docker build server -t coffee-prediction-server:latest
```

To start the server for the first time, use the following command:
```
docker run -it --network coffee-rating-prediction_devcontainer_default --hostname coffee_server --name coffee-prediction-server coffee-prediction-server:latest
```

To restart the server, run the following:
```
docker start -i coffee-prediction-server
```

Then run the following to test the server:
```
python test_predict.py
```
