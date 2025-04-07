import typing
import pickle
import pandas as pd


def data_to_xy(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data to features and target

    :param data: data
    :param target: name of target column
    :return: x, y
    """
    x = data.drop(target, axis=1)
    y = data[target]
    if isinstance(y, pd.Series):
        y = y.to_frame()
    return x, y


def xy_to_data(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate features and target

    :param x: features
    :param y: target
    :return: concatenated data
    """
    return pd.concat((x, y), axis=1)


def save(fn: str, data: typing.Any):
    """
    Save data to file

    :param fn: destination filename
    :param data: data to save
    """
    with open(fn, 'wb') as f:
        pickle.dump(data, f)


def read(fn: str) -> typing.Any:
    """
    Read data from file

    :param fn: source filename
    :return: data from file
    """
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data
