import pandas as pd


def data_to_xy(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = data.drop(target, axis=1)
    y = data[target]
    if isinstance(y, pd.Series):
        y = y.to_frame()
    return x, y
