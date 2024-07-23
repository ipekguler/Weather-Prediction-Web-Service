import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import pandas as pd
import mlflow
from flask import Flask, request, jsonify

best_run_id = os.environ["RUN_ID"]
bucket_name = os.environ["BUCKET"]


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def init_artifacts():
    ss = load_pickle("./artifacts/ss.pkl")
    oe = load_pickle("./artifacts/oe.pkl")
    le = load_pickle("./artifacts/le.pkl")

    return ss,oe,le

def prepare_features(weather, ss, oe):
    weather = pd.json_normalize(weather)
    categorical = ['Cloud Cover', 'Season', 'Location']
    numerical = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    weather.loc[:, numerical] = ss.transform(weather.loc[:, numerical])
    new_cols = ["Cloud Cover_clear","Cloud Cover_cloudy","Cloud Cover_overcast","Cloud Cover_partly cloudy","Season_Autumn","Season_Spring","Season_Summer","Season_Winter", "Location_coastal","Location_inland","Location_mountain"]
    weather[new_cols] = oe.transform(weather[categorical])
    weather.drop(columns=categorical, inplace=True)
    return weather

def predict(features, le, model):
    preds = model.predict(features)
    act_preds = le.inverse_transform(preds)
    return act_preds[0]

app = Flask('weather-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    weather = request.get_json()

    try:
        ss,oe,le = init_artifacts()
    except:
        mlflow.artifacts.download_artifacts(artifact_uri=f's3://{bucket_name}/4/{best_run_id}/artifacts/preprocessor/', dst_path="./artifacts")
        ss,oe,le = init_artifacts()
    model = mlflow.xgboost.load_model(f's3://{bucket_name}/4/{best_run_id}/artifacts/model/')

    weather = prepare_features(weather, ss, oe)
    pred = predict(weather, le, model)

    result = {
        "weather": pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)