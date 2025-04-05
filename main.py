import time
import argparse

from src.data_provider import DataProvider
from src.data_collector import DataCollector, data_to_xy
from src.EDA import process_na

TARGET = 'WITH_PAID'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to CSV file with dataset')
    return parser.parse_args()


def main():
    args = get_args()

    raw_data_path = args.dataset
    data_provider = DataProvider(raw_data_path)

    data_collector = DataCollector()
    while True:
        data = data_provider.get_batch()
        data = process_na(data)['result']
        x, y = data_to_xy(data, TARGET)
        data_collector.add(x, y)
        print(data_collector.x)
        time.sleep(3)


main()
