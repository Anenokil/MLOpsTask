import os
import time
import argparse
import logging
import yaml
from sklearn.tree import DecisionTreeClassifier

from src.data_provider import DataProvider
from src.data_collector import data_to_xy
from src.EDA_and_preprocessing import process
from src.model import Model

TARGET = 'WITH_PAID'
TIME_STAMP = 'INSR_BEGIN'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to YAML config')
    parser.add_argument('--dataset', help='Path to CSV file with dataset')
    parser.add_argument('--log_dir', help='Path to folder with logs')
    return parser.parse_args()


def read_config(args: argparse.Namespace):
    fn = args.config
    if not fn:
        return

    with open(fn, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        if k in args.__dict__ and args.__dict__[k] is None:
            args.__dict__[k] = v


def init_logger(log_dir: str):
    logs = []
    for fn in os.listdir(log_dir):
        name, ext = os.path.splitext(fn)
        if ext != '.log':
            continue
        logs.append(int(name))

    log_idx = 1 if not logs else max(logs) + 1
    log_fn = os.path.join(log_dir, f'{log_idx:05}.log')

    logging.basicConfig(filename=log_fn, level=logging.INFO, format='%(asctime)s - %(message)s')


def log_data_quality(data_quality: dict[str, ...]):
    na_by_col = data_quality['na_by_col'].tolist()
    rows_with_na = data_quality['rows_with_na']
    logging.info(f'na_by_col: {na_by_col}')
    logging.info(f'rows_with_na: {100 * rows_with_na}%')


def main():
    args = get_args()
    read_config(args)
    # for k, v in args.__dict__.items():
    #     print(f'{k} = {v}')

    init_logger(args.log_dir)

    raw_data_path = args.dataset
    data_provider = DataProvider(raw_data_path, TIME_STAMP)

    model = Model(DecisionTreeClassifier())

    while True:
        data = data_provider.get_batch()
        if data.empty:
            break
        logging.info(f'Get {data.shape[0]} samples')
        processed_data, stat = process(data)
        log_data_quality(stat['na'])
        logging.info(f'Keep {processed_data.shape[0]} samples')
        x, y = data_to_xy(processed_data, TARGET)
        if model.is_fit():
            print(model.eval())
        model.fit(x, y)
        time.sleep(3)


main()
