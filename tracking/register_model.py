import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

HPO_EXPERIMENT_NAME = "weather-model-select"
EXPERIMENT_NAME = "weather-best-model"
XGBC_PARAMS = ["learning_rate", "max_depth", "n_estimators", "random_state"]


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        for param in XGBC_PARAMS:
            try:
                params[param] = int(params[param])
            except:
                params[param] = float(params[param])

        XGBC = XGBClassifier(**params)
        XGBC.fit(X_train, y_train)

        val_f1 = f1_score(y_val, XGBC.predict(X_val), average="weighted")
        mlflow.log_metric("val_f1", val_f1)
        test_f1 = f1_score(y_test, XGBC.predict(X_test), average="weighted")
        mlflow.log_metric("test_f1", test_f1)

@click.command()
@click.option(
    "--data_path",
    help="Location where the processed weather data was saved."
)
@click.option(
    "--top_n",
    default=5,
    help="Number of top models that need to be evaluated to decide which one to promote."
)
def register_best_model(data_path: str, top_n: int):

    client = MlflowClient()

    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.f1_score DESC"]
    )

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_f1 DESC"]
    )[0]

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="weather-best-model")


if __name__ == '__main__':
    register_best_model()
