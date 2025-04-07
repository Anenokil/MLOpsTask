import pandas as pd

from src.utils import read, save


class DataProvider:
    def __init__(self, path_to_raw_data: str, time_stamp: str, path_to_save: str):
        """
        DataProvider emulates a streaming data source

        :param path_to_raw_data: path to data (CSV-file)
        :param time_stamp: name of column with time stamps
        :param path_to_save: path to file with DataProvider state
        """
        self.data = pd.read_csv(path_to_raw_data)
        self.time_stamp = time_stamp
        self.path_to_save = path_to_save
        self.i = 0

        self.__load_state()

    def __load_state(self):
        try:
            self.i = read(self.path_to_save)
        except FileNotFoundError:
            pass

    def __save_state(self):
        save(self.path_to_save, self.i)

    def get_day_data(self) -> pd.DataFrame:
        """
        Generate one-day data

        :return: data
        """
        date = self.data.loc[self.i, self.time_stamp]
        day_data = self.data[self.data[self.time_stamp] == date]
        self.i += day_data.shape[0]

        self.__save_state()

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

        self.__save_state()

        return batch
