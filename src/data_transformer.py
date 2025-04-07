import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataTransformer:
    def __init__(self, na_method='drop', ctg_method='ohe'):
        """
        DataTransformer processes data

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

    def __process_na(self):
        """
        Process missing values
        """
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
        categorical_cols = self.data.dtypes[self.data.dtypes == 'object'].index.tolist()
        if self.ctg_method == 'drop':
            self.data = self.data.drop(columns=categorical_cols)
        elif self.ctg_method == 'ohe':  # TODO
            # Update categories list
            ohe = OneHotEncoder()
            encoded_ctg = ohe.fit_transform(self.data[categorical_cols])
            # Concatenate non-categorical and encoded categorical features
            noncategorical = self.data.drop(categorical_cols, axis=1)
            encoded_ctg = pd.DataFrame(encoded_ctg.toarray(), columns=ohe.get_feature_names_out(),
                                       dtype=int, index=noncategorical.index)
            self.data = pd.concat([noncategorical, encoded_ctg], axis=1)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data

        :param df: data
        :return: processed data
        """
        self.data = df.copy()

        # Process missing values
        self.__process_na()
        # Process categorical features
        self.__process_ctg()

        return self.data
