import typing
import pandas as pd
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn import tree

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
        self.model = None  # Best model selected by self.selector
        self.selector = GridSearchCV(model, param_grid)
        self.path_to_save = path_to_save

        self.__load_state()

    def __load_state(self):
        try:
            self.data, self.transformer, self.model, self.selector = read(self.path_to_save)
        except FileNotFoundError:
            pass

    def __save_state(self):
        save(self.path_to_save, [self.data, self.transformer, self.model, self.selector])

    def fit(self, new_x: pd.DataFrame, new_y: pd.DataFrame):
        self.data.add(new_x, new_y)
        x, y = self.data.get()
        x, y = self.transformer.prepare_train(x, y)
        self.selector.fit(x, y)
        self.model = self.selector.best_estimator_

        self.__save_state()

        fig, axes = plt.subplots(nrows=1, ncols=self.model.n_estimators, figsize=(10, 2), dpi=900)
        for index in range(0, self.model.n_estimators):
            tree.plot_tree(self.model.estimators_[index],
                           filled=True,
                           ax=axes[index])

            axes[index].set_title('Estimator: ' + str(index), fontsize=11)
        fig.savefig('best_model.png')

    def refit(self, x: pd.DataFrame, y: pd.DataFrame):
        x, y = self.transformer.prepare_train(x, y)
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x, _ = self.transformer.prepare_pred(x)
        y = pd.DataFrame({'predicted': self.model.predict(x)})
        return x, y

    def eval(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        x, y = self.transformer.prepare_pred(x, y)
        return self.model.score(x, y)

    def is_fit(self) -> bool:
        return self.model is not None
