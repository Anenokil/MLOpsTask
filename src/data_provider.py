import pandas as pd


class DataProvider:
    def __init__(self, path_to_raw_data: str, target='WITH_PAID', time_stamp='INSR_BEGIN'):
        """
        DataProvider emulates a streaming data source

        :param path_to_raw_data: path to data (CSV-file)
        :param target: target column name
        :param time_stamp: name of column with time stamps
        """
        self.target = target
        self.time_stamp = time_stamp
        self.data = pd.read_csv(path_to_raw_data)
        self.i = 0

    def get_day_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate one-day data

        :return: data (x, y)
        """
        date = self.data.loc[self.i, self.time_stamp]
        day_data = self.data[self.data[self.time_stamp] == date]
        self.i += day_data.shape[0]

        x = day_data.drop(self.target, axis=1)
        y = day_data[self.target]
        return x, y

    def get_batch(self, batch_size=50) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate fixed amount of data

        :param batch_size: Amount of data
        :return: data (x, y)
        """
        start = self.i
        end = self.i + batch_size - 1
        self.i += batch_size

        x = self.data.loc[start:end].drop(self.target, axis=1)
        y = self.data.loc[start:end, self.target]
        return x, y
