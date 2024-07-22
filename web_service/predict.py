import os
import pickle
import pandas as pd
import mlflow
import boto3
from flask import Flask, request, jsonify

best_run_id = "d77afdb82bc248b4a23a36b83533b582"

s3 = boto3.resource(
    's3',
    region_name='us-east-1',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    endpoint_url='http://localstack:4566'
)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

model = mlflow.xgboost.load_model(f's3://weather-model-bucket/4/{best_run_id}/artifacts/best-model/model/')

def prepare_features(weather):

    ss = load_pickle("./ss.bin")
    oe = load_pickle("./oe.bin")

    weather = pd.json_normalize(weather)
    categorical = ['Cloud Cover', 'Season', 'Location']
    numerical = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    weather.loc[:, numerical] = ss.transform(weather.loc[:, numerical])
    new_cols = ["Cloud Cover_clear","Cloud Cover_cloudy","Cloud Cover_overcast","Cloud Cover_partly cloudy","Season_Autumn","Season_Spring","Season_Summer","Season_Winter", "Location_coastal","Location_inland","Location_mountain"]
    weather[new_cols] = oe.transform(weather[categorical])
    weather.drop(columns=categorical, inplace=True)
    return weather

def predict(features):
    #features = pd.DataFrame(features, index=[0])
    le = load_pickle("./le.bin")

    preds = model.predict(features)
    act_preds = le.inverse_transform(preds)
    return act_preds[0]

app = Flask('weather-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    weather = request.get_json()
    

    weather = prepare_features(weather)
    pred = predict(weather)

    result = {
        "weather": pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)