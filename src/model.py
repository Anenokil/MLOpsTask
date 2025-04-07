import typing
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.utils import read, save
from src.data_collector import DataCollector
from src.data_transformer import DataTransformer


class ModelPipeline:
    def __init__(self, transformer: DataTransformer, model, param_grid: dict[str, typing.Any], path_to_save: str):
        """
        ModelPipeline prepares data and then trains, evaluates, and validates model

        :param transformer: class to prepare data
        :param model: ML model from sklearn
        :param param_grid: parameters for grid search
        :param path_to_save: path to file with ModelPipeline state
        """
        self.data = DataCollector()
        self.transformer = transformer
        self.selector = GridSearchCV(model, param_grid)
        self.path_to_save = path_to_save

        self.__load_state()

    def __load_state(self):
        try:
            self.selector = read(self.path_to_save)
        except FileNotFoundError:
            pass

    def __save_state(self):
        save(self.path_to_save, self.selector)

    def fit(self, new_x: pd.DataFrame, new_y: pd.DataFrame):
        self.data.add(new_x, new_y)
        x, y = self.data.get()
        x, y = self.transformer.prepare_train(x, y)
        self.selector.fit(x, y)

        self.__save_state()

    def predict(self, x: pd.DataFrame) -> typing.Any:
        return self.selector.predict(x)

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        x, y = self.transformer.prepare_pred(x, y)
        return self.selector.score(x, y)

    def is_fit(self) -> bool:
        return self.data.x is not None
