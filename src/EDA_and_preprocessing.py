import pandas as pd


def process_na(df: pd.DataFrame) -> dict[str, ...]:
    """
    Process missed values in data

    :param df: data
    :return: dict with keys: 'result' - processed data, 'stat': percentage of missed values
    """
    stat = df.isna().sum(axis=0) / df.shape[0]
    new_df = df.dropna()
    return {'result': new_df, 'stat': stat}
