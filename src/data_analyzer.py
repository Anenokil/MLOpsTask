import typing
import pandas as pd


class DataAnalyzer:
    def __init__(self):
        """
        DataAnalyzer analyzes data
        """
        self.data = None
        self.stat = {}

    def __analyze_na(self):
        """
        Collect statistics on missing values
        """
        na_by_col = self.data.isna().sum(axis=0) / self.data.shape[0]
        na_by_row = self.data.isna().sum(axis=1)
        rows_with_na = na_by_row[na_by_row != 0].shape[0] / self.data.shape[0]
        self.stat['na'] = {'na_by_col': na_by_col, 'rows_with_na': rows_with_na}

    def __analyze_col_stats(self):
        """
        Collect statistics on completed values
        """
        categorical_cols = self.data.dtypes[self.data.dtypes == 'object'].index.tolist()
        categorical = self.data[categorical_cols]
        noncategorical = self.data.drop(categorical_cols, axis=1)
        self.stat['ctg_modes'] = categorical.mode(axis=0).to_numpy()[0]
        self.stat['num_medians'] = noncategorical.median()
        self.stat['num_means'] = noncategorical.mean()
        self.stat['num_vars'] = noncategorical.var()

    def analyze(self, df: pd.DataFrame) -> dict[str, typing.Any]:
        """
        Analyze data

        :param df: data
        :return: dict with information about data. Dict contains 'na' and 'ctg' keys
        """
        self.data = df.copy()
        self.stat = {}

        # Analyze missing values
        self.__analyze_na()
        self.__analyze_col_stats()

        return self.stat
