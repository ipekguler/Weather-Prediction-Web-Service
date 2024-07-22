import requests
import json

weather = {
    "Temperature": 10,
    "Humidity": 50,
    "Wind Speed": 9,
    "Precipitation (%)": 71,
    "Cloud Cover": "partly cloudy",
    "Atmospheric Pressure": 990,
    "UV Index": 2,
    "Season": "Winter",
    "Visibility (km)": 10,
    "Location": "inland"
}

url = 'http://myapp:9696/predict'
response = requests.post(url, json=weather)
print(response.text)