import typing
import os
import time
import argparse
import logging
import yaml
from sklearn.tree import DecisionTreeClassifier

from src.data_provider import DataProvider
from src.data_collector import data_to_xy
from src.EDA_and_preprocessing import DataTransformer
from src.model import Model

TARGET = 'WITH_PAID'  # Target column in data
TIME_STAMP = 'INSR_BEGIN'  # Column with time stamps


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to YAML config')
    parser.add_argument('-d', '--dataset', help='Path to CSV file with dataset')
    parser.add_argument('-l', '--log_dir', help='Path to folder with logs')
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true', required=False)
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


def log_data_quality(data_quality: dict[str, typing.Any]):
    na_by_col = data_quality['na_by_col'].tolist()
    rows_with_na = data_quality['rows_with_na']
    logging.info(f'na_by_col: {na_by_col}')
    logging.info(f'rows_with_na: {100 * rows_with_na}%')


def main():
    # Get args from console
    args = get_args()
    # Read args from config
    read_config(args)

    # Initialize logger
    init_logger(args.log_dir)

    # Initialize data provider
    raw_data_path = args.dataset
    data_provider = DataProvider(raw_data_path, TIME_STAMP)

    # Initialize data transformer
    data_processor = DataTransformer(na_method='median-mode', ctg_method='drop')

    # Initialize ML model
    model = Model(DecisionTreeClassifier())

    if args.verbose:
        print('Start')
    while True:
        # Get data batch
        data = data_provider.get_batch()
        if data.empty:
            break
        if args.verbose:
            print('Receive new data')
        logging.info(f'Get {data.shape[0]} samples')
        # Analyze and preprocess data
        processed_data, stat = data_processor.process(data)
        log_data_quality(stat['na'])
        logging.info(f'Keep {processed_data.shape[0]} samples')
        x, y = data_to_xy(processed_data, TARGET)
        # Evaluate model
        if model.is_fit():
            if args.verbose:
                print('Evaluate model')
            print(model.eval(x, y))
        # Train model
        if args.verbose:
            print('Train model')
        model.fit(x, y)
        # Emulate delay between data arrivals
        time.sleep(3)
    if args.verbose:
        print('End')


main()
