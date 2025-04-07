import typing
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.data_collector import DataCollector
from src.data_transformer import DataTransformer


class ModelPipeline:
    def __init__(self, transformer: DataTransformer, model, params: dict[str, typing.Any]):
        """
        ModelPipeline prepares data and then trains, evaluates, and validates model

        :param transformer: class to prepare data
        :param model: ML model from sklearn
        :param params: parameters for grid search
        """
        self.data = DataCollector()
        self.transformer = transformer
        self.selector = GridSearchCV(model, params)

    def fit(self, new_x: pd.DataFrame, new_y: pd.DataFrame):
        self.data.add(new_x, new_y)
        x, y = self.data.get()
        x, y = self.transformer.prepare_train(x, y)
        self.selector.fit(x, y)

    def predict(self, x: pd.DataFrame) -> typing.Any:
        return self.selector.predict(x)

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        x, y = self.transformer.prepare_pred(x, y)
        return self.selector.score(x, y)

    def is_fit(self) -> bool:
        return self.data.x is not None
