import typing
import pandas as pd

from src.data_collector import DataCollector


class Model:
    def __init__(self, model):
        """
        Model trains, evaluates, and validates model

        :param model: ML model from sklearn
        """
        self.model = model
        self.data = DataCollector()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.data.add(x, y)
        self.model.fit(*self.data.get())

    def predict(self, x: pd.DataFrame) -> typing.Any:
        return self.model.predict(x)

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        return self.model.score(x, y)

    def is_fit(self) -> bool:
        return self.data.x is not None
