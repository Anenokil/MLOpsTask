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
