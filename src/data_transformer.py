import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.utils import data_to_xy


class DataTransformer:
    def __init__(self, na_method='drop', ctg_method='ohe'):
        """
        DataTransformer processes data

        :param na_method: how to process missing values:
            'drop' for remove lines with missing values;
            'median-mode' for replace them with median (for numeric) or mode (for categorical)
        :param ctg_method: how to process categorical features:
            'ohe' for one-hot encoding
        """
        assert na_method in ['drop', 'median-mode']
        assert ctg_method in ['ohe']

        self.na_method = na_method
        self.ctg_method = ctg_method

        self.ohe = None
        self.ohe_categories = None

    def __process_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process missing values
        
        :param data: data
        :return: processed data
        """
        if self.na_method == 'drop':
            data = data.dropna()
        elif self.na_method == 'median-mode':
            categorical_cols = data.dtypes[data.dtypes == 'object'].index.tolist()
            noncategorical_cols = data.dtypes[data.dtypes != 'object'].index.tolist()
            modes = data[categorical_cols].mode().to_numpy()[0].tolist()
            medians = data[noncategorical_cols].median().tolist()
            placeholders = dict(zip(categorical_cols + noncategorical_cols, modes + medians))
            data = data.fillna(value=placeholders)

        return data

    def __process_ctg(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features

        :param data: data
        :return: processed data
        """
        if self.ctg_method == 'ohe':
            # Encode categorical features
            categorical_cols = data.dtypes[data.dtypes == 'object'].index.tolist()
            self.ohe = OneHotEncoder()
            encoded_ctg = self.ohe.fit_transform(data[categorical_cols])
            # Save categories
            self.ohe_categories = self.ohe.categories_
            # Concatenate non-categorical and encoded categorical features
            noncategorical = data.drop(categorical_cols, axis=1)
            encoded_ctg = pd.DataFrame(encoded_ctg.toarray(), columns=self.ohe.get_feature_names_out(),
                                       dtype=int, index=noncategorical.index)
            data = pd.concat([noncategorical, encoded_ctg], axis=1)

        return data

    def __rm_unknown_ctg(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing unknown categorical values from the data

        :param data: data
        :return: processed data
        """
        # Get columns with categorical values
        categorical_cols = data.dtypes[data.dtypes == 'object'].index.tolist()
        categorical = data[categorical_cols]
        # Search unknown categories in columns
        unknown = []
        for i in range(categorical.shape[1]):
            col_name = categorical_cols[i]
            col = data[col_name]
            unknown.append(col.apply(lambda val: val not in self.ohe_categories[i]))
        unknown = pd.concat(unknown, axis=1).any(axis=1)
        # Get rows containing unknown categories
        rows_with_unknown = np.where(unknown, unknown.index, -1)
        rows_with_unknown = set(rows_with_unknown)
        rows_with_unknown.discard(-1)
        # Drop these rows
        data = data.drop(index=list(rows_with_unknown), axis=0)
        # Encode categorical features
        encoded_ctg = self.ohe.transform(data[categorical_cols])
        # Concatenate non-categorical and encoded categorical features
        noncategorical = data.drop(categorical_cols, axis=1)
        encoded_ctg = pd.DataFrame(encoded_ctg.toarray(), columns=self.ohe.get_feature_names_out(),
                                   dtype=int, index=noncategorical.index)
        data = pd.concat([noncategorical, encoded_ctg], axis=1)

        return data

    def prepare_train(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process data to prepare it for training

        :param x: features
        :param y: target
        :return: processed (x, y)
        """
        data = pd.concat((x, y), axis=1)

        # Process missing values
        data = self.__process_na(data)
        # Process categorical features
        data = self.__process_ctg(data)

        x, y = data_to_xy(data, y.columns)
        return x, y

    def prepare_pred(self, x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process data to prepare it for making prediction

        :param x: features
        :param y: target
        :return: processed (x, y)
        """
        data = pd.concat((x, y), axis=1)

        # Process missing values
        data = self.__process_na(data)
        # Process categorical features
        if self.ctg_method == 'ohe':
            data = self.__rm_unknown_ctg(data)

        x, y = data_to_xy(data, y.columns)
        return x, y
