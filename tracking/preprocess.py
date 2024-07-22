import os
import pickle
import click
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df = df[(df['Humidity'] <= 100) & (df['Precipitation (%)'] <= 100)]

    return df


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw weather data is saved."
)
@click.option(
    "--dest_path",
    help="Location where the output will be saved."
)
def data_prep(raw_data_path: str, dest_path: str, dataset: str = "weather_classification_data"):
    df = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}.csv")
    )
    
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

    dump_pickle(le, os.path.join(dest_path, "le.pkl"))
    dump_pickle(ss, os.path.join(dest_path, "ss.pkl"))
    dump_pickle(oe, os.path.join(dest_path, "oe.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))


if __name__ == '__main__':
    data_prep()