import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def process_na(df: pd.DataFrame) -> dict[str, ...]:
    """
    Process missed values - removes them

    :param df: data
    :return: dict with keys: 'result' - processed data, 'stat': percentage of missed values
    """
    na_by_col = df.isna().sum(axis=0) / df.shape[0]
    na_by_row = df.isna().sum(axis=1)
    rows_with_na = na_by_row[na_by_row != 0].shape[0] / df.shape[0]
    stat = {'na_by_col': na_by_col, 'rows_with_na': rows_with_na}

    res = df.dropna()

    return {'result': res, 'stat': stat}


def process_ctg(df: pd.DataFrame) -> dict[str, ...]:
    """
    Process categorical features - encode them

    :param df: data
    :return: dict with keys: 'result' - processed data, 'stat': names of categorical features
    """
    ohe = OneHotEncoder()
    categorical = df.dtypes[df.dtypes == 'object'].index.tolist()
    noncategorical = df.drop(categorical, axis=1)
    encoded = ohe.fit_transform(df[categorical])
    encoded = pd.DataFrame(encoded.toarray(), columns=ohe.get_feature_names_out(),
                           dtype=int, index=noncategorical.index)
    res = pd.concat([noncategorical, encoded], axis=1)
    return {'result': res, 'stat': categorical}


def process(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, ...]]:
    """
    Process data

    :param df: data
    :return: processed data and dict with information about data
    """
    stat = {}
    data = df.copy()

    res = process_na(data)
    data, stat['na'] = res['result'], res['stat']

    res = process_ctg(data)
    data, stat['ctg'] = res['result'], res['stat']

    return data, stat
