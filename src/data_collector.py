import pandas as pd


class DataCollector:
    def __init__(self, x: pd.DataFrame = None, y: pd.DataFrame = None):
        """
        DataCollector aggregates received data

        :param x: x
        :param y: y
        """
        self.x = x
        self.y = y

    def add(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Add new data to collector

        :param x: x
        :param y: y
        """
        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = pd.concat([self.x, x], axis=0)
            self.y = pd.concat([self.y, y], axis=0)

    def get(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get stored data

        :return: data (x, y)
        """
        return self.x, self.y
