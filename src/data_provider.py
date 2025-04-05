import pandas as pd


class DataProvider:
    def __init__(self, path_to_raw_data: str, time_stamp='INSR_BEGIN'):
        """
        DataProvider emulates a streaming data source

        :param path_to_raw_data: path to data (CSV-file)
        :param time_stamp: name of column with time stamps
        """
        self.data = pd.read_csv(path_to_raw_data)
        self.time_stamp = time_stamp
        self.i = 0

    def get_day_data(self) -> pd.DataFrame:
        """
        Generate one-day data

        :return: data
        """
        date = self.data.loc[self.i, self.time_stamp]
        day_data = self.data[self.data[self.time_stamp] == date]
        self.i += day_data.shape[0]

        return day_data

    def get_batch(self, batch_size=50) -> pd.DataFrame:
        """
        Generate fixed amount of data

        :param batch_size: Amount of data
        :return: data
        """
        start = self.i
        end = self.i + batch_size - 1
        batch = self.data.loc[start:end]
        self.i += batch_size

        return batch
