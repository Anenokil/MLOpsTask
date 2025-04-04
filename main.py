import time
import argparse

from src.data_provider import DataProvider
from src.data_collector import DataCollector


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
        x, y = data_provider.get_day_data()
        data_collector.add(x, y)
        print(data_collector.x)
        time.sleep(2)


main()
