import pickle
import mlflow
import os

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


mlflow.end_run()

HPO_EXPERIMENT_NAME = "weather-xgbc-models"
EXPERIMENT_NAME = "weather-xgbc-best-model"
XGBC_PARAMS = ["learning_rate", "max_depth", "n_estimators", "random_state"]

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params):

    with mlflow.start_run():
        for param in XGBC_PARAMS:
            try:
                params[param] = int(params[param])
            except:
                params[param] = float(params[param])

        XGBC = XGBClassifier(params)
        XGBC.fit(X_train, y_train)

        mlflow.xgboost.log_model(XGBC, artifact_path="model")
        val_f1 = f1_score(y_val, XGBC.predict(X_val), average="weighted")
        mlflow.log_metric("val_f1", val_f1)
        test_f1 = f1_score(y_test, XGBC.predict(X_test), average="weighted")
        mlflow.log_metric("test_f1", test_f1)


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

    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=int(os.environ["NUM_TRIALS"]),
        order_by=["metrics.f1_score DESC"]
    )

    for run in runs:
        train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, run.data.params)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=int(os.environ["NUM_TRIALS"]),
        order_by=["metrics.test_f1 DESC"]
    )[0]

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="weather-best-model")

    loaded_model = mlflow.xgboost.load_model(model_uri)
    mlflow.xgboost.log_model(loaded_model, artifact_path='best-model/model')

    with open('./weather_prediction_project/artifacts/le.pkl', 'wb') as f_out:
        pickle.dump(le, f_out)

    with open('./weather_prediction_project/artifacts/ss.pkl', 'wb') as f_out:
        pickle.dump(ss, f_out)

    with open('./weather_prediction_project/artifacts/oe.pkl', 'wb') as f_out:
        pickle.dump(oe, f_out)

    mlflow.log_artifacts('./weather_prediction_project/artifacts/', artifact_path='preprocessor', run_id=run_id)

    print(f"best model's run id: {run_id}")
    #print(s3.list_objects_v2(Bucket="weather-model-bucket"))


