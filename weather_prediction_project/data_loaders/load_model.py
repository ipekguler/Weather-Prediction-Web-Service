import mlflow
from mlflow.tracking import MlflowClient

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

mlflow.set_tracking_uri("http://mlflow:5000")

@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    model_name = "weather-xgbx-best-model"
    model_version = "1"

    model_uri=f"models:/{model_name}/{model_version}"

    model = mlflow.pyfunc.load_model(model_uri)

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
