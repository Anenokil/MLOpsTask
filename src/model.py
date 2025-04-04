import pandas as pd
from src.data_collector import DataCollector


class Model:
    def __init__(self, model):
        self.model = model
        self.data = DataCollector()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.data.add(x, y)
        self.model.fit(*self.data.get())

    def predict(self, x: pd.DataFrame):
        return self.model.predict(x)

    def eval(self):
        x, y = self.data.get()
        if x is None:
            return None
        return self.model.score(x, y)
