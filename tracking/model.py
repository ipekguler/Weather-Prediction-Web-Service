import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import click
import mlflow
import numpy as np
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("weather-model-select")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    help="Location where the processed weather data was saved."
)
@click.option(
    "--num_trials",
    default=25,
    help="The number of parameter evaluations for the optimizer to explore."
)
def fit_models(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

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
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    fit_models()