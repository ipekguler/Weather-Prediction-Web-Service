# Weather Prediction Service

Final project for [DataTalksClub's MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

## Dataset

Weather forecasting is crucial for various industries and everyday planning. In this project, we aim to develop a machine learning model that predicts the type of weather based on historical weather data. The dataset used for this project contains various meteorological features such as temperature, humidity, wind speed, and pressure, which will be utilized to classify weather types into categories such as sunny, rainy, cloudy, etc.

The dataset includes:

* Features: Various meteorological measurements including temperature, humidity, wind speed, pressure, etc.
* Target: Weather type categories (e.g., sunny, rainy, cloudy) based on the provided weather data.

Link to the dataset: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data

## Service Usage

A web service is used to get weather predictions. The user sends a request to the web service with data such as temperature, humidity, wind speed, and pressure and the web service responds with the predicted weather type for that data.

## Project Overview

* **Cloud**: The project uses localstack to use AWS S3. Sklearn preprocessors and models are uploaded it to S3 for later use to make predictions inside the web service. 
* **Experiment tracking and model registry**: Experiments are tracked using MLFlow. Parameters, artifacts and models are logged while experimenting with models. The best performing model is selected and registered to MLFlow model registry.
* **Workflow orchestration**: Loading data, preprocessing, model fitting, hyperparameter tuning, experiment tracking and model registry are all orchestrated using Mage.
* **Model deployment**: The model is deployed as a web service integrated with MLFlow using Flask and gunicorn. The model deployment code is containerized and could be deployed to the cloud.
* **Model monitoring**: Evidently is used to calculate and report metrics. A script is writted to simulate new incoming data and the data is sent to PostgreSQL. A Grafana dashboard is created to see metrics for incoming data such as number of drifted columns, prediction drift and missing data.
* **Reproducibility**: All the main components are completely containerized. Components run locally for experiments have requirements and Pipfiles. The instructions to run the project are stated below.

## Usage

* ### Project setup

Clone the repository.

```
$ git clone https://github.com/ipekguler/Weather-Prediction-Web-Service.git
```

Create an external network called "weather-net". This is required to put the web service in the same network as Localstack S3 so it can load the model and artifacts to make predictions.

```
$ docker network create "weather-net"
```

* ### Developing & saving the model and preprocessors

Start running Mage, MLFlow and Localstack. Mage will start the project "weather_prediction_project".

```
$ cd <repo-base-dir>
$ docker-compose up
```

On a browser, go to the Mage UI at http://localhost:6789 and run the pipeline called "weather_xgbc_pipeline" until the end. This pipeline will take the Weather Type Classification Data from the local file system, preprocess it, fit models, do hyperparameter tuning and choose the best model while tracking every step on MLFlow. Models are saved to Localstack S3.

After the pipeline is done, scroll down to copy the run id for the best model. You will need it to download the model from Localstack S3 when buiding the web service in the next step.

You can modify the variables in the .env file to customize model selection process such as:

* TOP_N: determines how many initial model fits are to be made.
* NUM_TRIALS: determines the number of best performing models for final evaluation.

* ### Starting the web service

Start the web service:

```
$ cd <repo-base-dir>/web_service
$ docker-compose up
```
The service will start running and listening to the requests. "test.py" script in the web_service directory contains sample data that could be sent to the service to get weather prediction. You can use the same parameters or modify them.

Go to a seperate shell to send a POST request to the server with the default data.

```
$ docker exec -it web_service-myapp-1 sh
$ python test.py
```

The web service loads model and preprocessors from Localstack S3. If downloaded before, the service reads them from its file system. After making the request, you should get a response like the following:

```
"weather":"Cloudy"
```

* ### Monitoring

Set up the environment for monitoring.

```
$ conda create -n project python=3.11
$ conda activate project
$ pip install -r requirements.txt
```

Using the activated environment, go to jupyter notebook UI and launch "weather_prediction_evidently" notebook. Run this notebook to save the full and reference dataset we will need when calculating the metrics.

Start running Postgres, Adminer and Grafana.

```
$ cd <repo-base-dir>/monitoring
$ docker-compose up
```

"calculate_metrics.py" script is used to simulate new data flow. The script takes randomized portions of the data, treats it like incoming data and compares it to a reference dataset to calculate metrics such as prediction drift and missing data.

On a seperate shell, go to the monitoring directory, activate the conda project environment and run the script to get data flowing to Postgres and a pre-defined Grafana dashboard.

```
$ cd <repo-base-dir>/monitoring
$ conda activate project
$ python calculate_metrics.py
```

On a browser, go to the Grafana UI at http://localhost:3030 to observe the data quality dashboard. The default username and password are admin.