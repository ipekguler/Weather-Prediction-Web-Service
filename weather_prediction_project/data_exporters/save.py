import mlflow
import numpy as np
import pickle
import os
import boto3
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("weather-xgbc-models")

mlflow.end_run()

s3 = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    endpoint_url='http://localstack:4566'
)

s3.create_bucket(
    Bucket="weather-model-bucket", 
)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """

    le, ss, oe, X_train, y_train, X_test, y_test, X_val, y_val = data


    def objective (params):
        with mlflow.start_run():
            mlflow.log_params(params)
            clf=XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred=clf.predict(X_val)
            
            accuracy=accuracy_score(y_val, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            f1= f1_score(y_val, y_pred, average='weighted')
            mlflow.log_metric("f1_score", f1)
            precision = precision_score(y_val, y_pred, average='weighted')
            mlflow.log_metric("precision_score", precision)
            recall = recall_score(y_val, y_pred, average='weighted')
            mlflow.log_metric("recall_score", recall)

        loss = log_loss(y_val, clf.predict_proba(X_val))
        return {'loss': loss, 'status': STATUS_OK } 

    search_space = {
        "max_depth" : scope.int(hp.quniform('max_depth', 1, 10, 1)),
        "learning_rate": hp.quniform('learning_rate', 0.01, 0.3, 0.01),
        "n_estimators": scope.int(hp.quniform('n_estimators', 25, 125, 5)),
        "random_state": 42
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=int(os.environ["TOP_N"]),
        trials=Trials(),
        rstate=rstate
    )

    return le, ss, oe, X_train, y_train, X_test, y_test, X_val, y_val
