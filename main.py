import time
import argparse
from sklearn.tree import DecisionTreeClassifier

from src.data_provider import DataProvider
from src.data_collector import data_to_xy
from src.EDA_and_preprocessing import process
from src.model import Model

TARGET = 'WITH_PAID'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to CSV file with dataset')
    return parser.parse_args()


def main():
    args = get_args()

    raw_data_path = args.dataset
    data_provider = DataProvider(raw_data_path)

    model = Model(DecisionTreeClassifier())

    while True:
        data = data_provider.get_batch()
        data, stat = process(data)
        x, y = data_to_xy(data, TARGET)
        print(model.eval())
        model.fit(x, y)
        time.sleep(3)


main()
