import pandas as pd


def process_na(df: pd.DataFrame) -> dict[str, ...]:
    stat = df.isna().sum(axis=0) / df.shape[0]
    new_df = df.dropna()
    return {'result': new_df, 'stat': stat}
