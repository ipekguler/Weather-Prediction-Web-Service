# Weather Prediction Service

Final project for [DataTalksClub's MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

## Dataset

Weather forecasting is crucial for various industries and everyday planning. In this project, we aim to develop a machine learning model that predicts the type of weather based on historical weather data. The dataset used for this project contains various meteorological features such as temperature, humidity, wind speed, and pressure, which will be utilized to classify weather types into categories such as sunny, rainy, cloudy, etc.

The dataset includes:

* Features: Various meteorological measurements including temperature, humidity, wind speed, pressure, etc.
* Target: Weather type categories (e.g., sunny, rainy, cloudy) based on the provided weather data.

Download the dataset here: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data

## Service Usage

A web service is used to get weather predictions. The user sends a request to the web service with data such as temperature, humidity, wind speed, and pressure and the web service responds with the predicted weather type for that data.

## Project Overview

* **Cloud**: The project uses localstack to use AWS S3. Models are fitted through hyperparameter tuning and then uploaded it to S3 for later use to make predictions.
* **Experiment tracking and model registry**: Experiments are tracked using MLFlow. Parameters, artifacts and models are logged while experimenting with models. The best performing model is selected and registered to MLFlow model registry.
* **Workflow orchestration**: Loading data, preprocessing, model fitting, hyperparameter tuning, experiment tracking and model registry are all orchestrated using Mage.
* **Model deployment**: The model is deployed as a web service integrated with MLFlow using Flask and gunicorn. The model deployment code is containerized and could be deployed to the cloud.
* **Model monitoring**: Evidently is used to calculate and report metrics. A script is writted to simulate new incoming data and the data is sent to PostgreSQL. A Grafana dashboard is created to see metrics for incoming data such as number of drifted columns, prediction drift and missing data.
* **Reproducibility**: All the main components are completely containerized. Components run locally for experiments have requirements and Pipfiles. The instructions to run the project are stated below.

## Usage

* ### Developing & saving the model

Run the following prompt to start running Mage, MLFlow and Localstack. Mage will start the project "weather_prediction_project".

```
$ cd <repo-base-dir>
$ docker-compose up
```

On a browser, go to the Mage UI at http://localhost:6789 and run the pipeline called "weather_xgbc_pipeline" until the end. This pipeline will take the Weather Type Classification Data from the local file system, preprocess it, fit models, do hyperparameter tuning and choose the best model while tracking every step on MLFlow. Models are saved to Localstack S3.

After the pipeline is done, scroll down to copy the run id for the best model. You will need it to download the model from Localstack S3 when buiding the web service in the next step.

You can modify the following parameters in the pipeline to customize model selection process:

* num_trials: determines how many models fits are to be made
* top_n: determines the number of best performing models for final evaluation.

* ### Starting the web service

Run the following prompt to start the web service. The service will start running and listening to the requests.

```
$ cd <repo-base-dir>/web_service
$ docker-compose up
```

"test.py" script in the web_service directory contains sample data that could be sent to the service to get weather prediction. You can use the same parameters or modify them.

To send a POST request to the server with the default data, go to a seperate shell and run the following prompt.

```
$ docker exec -it web_service-myapp-1 sh
$ python test.py
```

You should get a response like the following:

```
"weather":"Cloudy"
```

* ### Monitoring

Run the following prompt to set up the environment.

```
$ conda create -n project python=3.11
$ conda activate project
$ pip install -r requirements.txt
```

Using the activated environment, go to jupyter notebook UI and launch "weather_prediction_evidently" notebook. Run this notebook to save the full and reference dataset we will need when calculating the metrics.

Run the following prompt to start running Postgres, Adminer and Grafana.

```
$ cd <repo-base-dir>/monitoring
$ docker-compose up
```

"calculate_metrics.py" script is used to simulate new data flow. The script takes randomized portions of the data, treats it like incoming data and compares it to a reference dataset to calculate metrics such as prediction drift and missing data.

On a seperate shell, activate the conda project environment and run the following prompt to get data flowing to Postgres and a pre-defined Grafana dashboard.

```
$ python calculate_metrics.py
```

On a browser, go to the Grafana UI at http://localhost:3030 to observe the data quality dashboard.