import pandas as pd


class DataCollector:
    def __init__(self):
        """
        DataCollector aggregates received data
        """
        self.x: pd.DataFrame | None = None
        self.y: pd.DataFrame | None = None

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
