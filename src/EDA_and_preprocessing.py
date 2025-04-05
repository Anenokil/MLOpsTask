import typing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataTransformer:
    def __init__(self, na_method='drop', ctg_method='ohe'):
        """
        DataProcessor analyzes and processes data

        :param na_method: how to process missing values:
            'drop' for remove lines with missing values;
            'median-mode' for replace them with median (for numeric) or mode (for categorical)
        :param ctg_method: how to process categorical features:
            'drop' for remove categorical columns;
            'ohe' for one-hot encoding
        """
        assert na_method in ['drop', 'median-mode']
        assert ctg_method in ['drop', 'ohe']

        self.na_method = na_method
        self.ctg_method = ctg_method

        self.data = None
        self.stat = {}

        self.categories = []

    def __process_na(self):
        """
        Process missing values
        """
        # Collect statistics on missing values
        na_by_col = self.data.isna().sum(axis=0) / self.data.shape[0]
        na_by_row = self.data.isna().sum(axis=1)
        rows_with_na = na_by_row[na_by_row != 0].shape[0] / self.data.shape[0]
        self.stat['na'] = {'na_by_col': na_by_col, 'rows_with_na': rows_with_na}

        # Process missing values
        if self.na_method == 'drop':
            self.data = self.data.dropna()
        elif self.na_method == 'median-mode':
            categorical_cols = self.data.dtypes[self.data.dtypes == 'object'].index.tolist()
            noncategorical_cols = self.data.dtypes[self.data.dtypes != 'object'].index.tolist()
            modes = self.data[categorical_cols].mode().to_numpy()[0].tolist()
            medians = self.data[noncategorical_cols].median().tolist()
            placeholders = dict(zip(categorical_cols + noncategorical_cols, modes + medians))
            self.data = self.data.fillna(value=placeholders)

    def __process_ctg(self):
        """
        Process categorical features
        """
        # Collect statistics on categorical features
        categorical_cols = self.data.dtypes[self.data.dtypes == 'object'].index.tolist()
        self.stat['ctg'] = categorical_cols

        # Process categorical features
        if self.ctg_method == 'drop':
            self.data = self.data.drop(columns=categorical_cols)
        elif self.ctg_method == 'ohe':  # TODO
            # Update categories list
            ohe = OneHotEncoder()
            ohe.fit(self.data[categorical_cols])
            if not self.categories:
                self.categories = ohe.categories_
            else:
                self.categories = [list(set(list(a) + list(b))) for a, b in zip(self.categories, ohe.categories_)]
            # Apply OHE
            ohe = OneHotEncoder(categories=self.categories)
            encoded_ctg = ohe.fit_transform(self.data[categorical_cols])
            # Concatenate non-categorical and encoded categorical features
            noncategorical = self.data.drop(categorical_cols, axis=1)
            encoded_ctg = pd.DataFrame(encoded_ctg.toarray(), columns=ohe.get_feature_names_out(),
                                       dtype=int, index=noncategorical.index)
            self.data = pd.concat([noncategorical, encoded_ctg], axis=1)

    def process(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, typing.Any]]:
        """
        Process data

        :param df: data
        :return: processed data and dict with information about data.
            Dict contains 'na', 'ctg' keys
        """
        self.data = df.copy()
        self.stat = {}

        # Process missing values
        self.__process_na()
        # Process categorical features
        self.__process_ctg()

        return self.data, self.stat
