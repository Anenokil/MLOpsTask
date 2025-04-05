import typing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def process_na(df: pd.DataFrame, method: str) -> dict[str, typing.Any]:
    """
    Process missing values

    :param df: data
    :param method: how to process missing values:
        'drop' for remove lines with missing values;
        'median-mode' for replace them with median (for numeric) or mode (for categorical)
    :return: dict with keys: 'result' - processed data, 'stat': percentage of missing values
    """
    assert method in ['drop', 'median-mode']

    # Collect statistics on missing values
    na_by_col = df.isna().sum(axis=0) / df.shape[0]
    na_by_row = df.isna().sum(axis=1)
    rows_with_na = na_by_row[na_by_row != 0].shape[0] / df.shape[0]
    stat = {'na_by_col': na_by_col, 'rows_with_na': rows_with_na}

    # Process missing values
    if method == 'drop':
        res = df.dropna()
    elif method == 'median-mode':
        categorical_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
        noncategorical_cols = df.dtypes[df.dtypes != 'object'].index.tolist()
        modes = df[categorical_cols].mode().to_numpy()[0].tolist()
        medians = df[noncategorical_cols].median().tolist()
        placeholders = dict(zip(categorical_cols + noncategorical_cols, modes + medians))
        res = df.fillna(value=placeholders)

    return {'result': res, 'stat': stat}


def process_ctg(df: pd.DataFrame) -> dict[str, typing.Any]:
    """
    Process categorical features - encode them

    :param df: data
    :return: dict with keys: 'result' - processed data, 'stat': names of categorical features
    """
    ohe = OneHotEncoder()
    categorical_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    noncategorical = df.drop(categorical_cols, axis=1)
    encoded_ctg = ohe.fit_transform(df[categorical_cols])
    encoded_ctg = pd.DataFrame(encoded_ctg.toarray(), columns=ohe.get_feature_names_out(),
                               dtype=int, index=noncategorical.index)
    res = pd.concat([noncategorical, encoded_ctg], axis=1)
    return {'result': res, 'stat': categorical_cols}


def process(df: pd.DataFrame, na_method='drop') -> tuple[pd.DataFrame, dict[str, typing.Any]]:
    """
    Process data

    :param df: data
    :param na_method: how to process missing values:
        'drop' for remove lines with missing values;
        'median-mode' for replace them with median (for numeric) or mode (for categorical)
    :return: processed data and dict with information about data
    """
    stat = {}
    data = df.copy()

    res = process_na(data, method=na_method)
    data, stat['na'] = res['result'], res['stat']

    res = process_ctg(data)
    data, stat['ctg'] = res['result'], res['stat']

    return data, stat
