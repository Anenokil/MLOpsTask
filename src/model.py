import typing
import pandas as pd

from src.data_collector import DataCollector
from src.data_transformer import DataTransformer


class ModelPipeline:
    def __init__(self, model, transformer: DataTransformer):
        """
        ModelPipeline prepares data and then trains, evaluates, and validates model

        :param model: ML model from sklearn
        :param transformer: class to prepare data
        """
        self.model = model
        self.data = DataCollector()
        self.transformer = transformer

    def fit(self, new_x: pd.DataFrame, new_y: pd.DataFrame):
        self.data.add(new_x, new_y)
        x, y = self.data.get()
        x, y = self.transformer.prepare_train(x, y)
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame) -> typing.Any:
        return self.model.predict(x)

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        x, y = self.transformer.prepare_pred(x, y)
        return self.model.score(x, y)

    def is_fit(self) -> bool:
        return self.data.x is not None
