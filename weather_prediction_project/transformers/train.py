import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    target = 'Weather Type'
    categorical = ['Cloud Cover', 'Season', 'Location']
    numerical = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']

    X = df.drop(columns=target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    ss=StandardScaler()
    X_train.loc[:, numerical] = ss.fit_transform(X_train.loc[:, numerical])
    X_test.loc[:, numerical] = ss.transform(X_test.loc[:, numerical])
    X_val.loc[:, numerical] = ss.transform(X_val.loc[:, numerical])

    oe=OneHotEncoder(sparse_output=False)
    oe.fit(X_train[categorical])
    new_cols = oe.get_feature_names_out()
    X_train[new_cols] = oe.transform(X_train[categorical])
    X_test[new_cols] = oe.transform(X_test[categorical])
    X_val[new_cols] = oe.transform(X_val[categorical])

    X_train.drop(columns=categorical, inplace=True)
    X_test.drop(columns=categorical, inplace=True)
    X_val.drop(columns=categorical, inplace=True)

    return le, ss, oe, X_train, y_train.tolist(), X_test, y_test.tolist(), X_val, y_val.tolist()


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
