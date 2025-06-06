import typing
import os
import time
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from memory_profiler import profile
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cosine

from src.utils import data_to_xy, xy_to_data
from src.data_provider import DataProvider
from src.data_analyzer import DataAnalyzer
from src.data_transformer import DataTransformer
from src.model import ModelPipeline

import warnings
warnings.filterwarnings('ignore')

TARGET = 'WITH_PAID'  # Target column in data
TIMESTAMPS = 'INSR_BEGIN'  # Column with timestamps
PATH_TO_DATA_PROVIDER_SAVES = os.path.join('.states', 'dp.pkl')  # Path to file with DataProvider saved state
PATH_TO_MODEL_PIPELINE_SAVES = os.path.join('.states', 'mp.pkl')  # Path to file with ModelPipeline saved state
PAUSE = 3  # Pause (in seconds) between data arrivals

if not os.path.exists('.states'):
    os.mkdir('.states')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to YAML config')
    parser.add_argument('-d', '--data', help='Path to CSV file with dataset')
    parser.add_argument('-o', '--out', help='Output path for inference')
    #parser.add_argument('-l', '--logs', help='Path to folder with logs', default='.logs')
    parser.add_argument('-m', '--mode', choices=['train', 'update', 'eval', 'inference', 'summary'], help='Action type')
    parser.add_argument('-n', '--n_iter', help='Number of training iterations. Set 0 to train on all data',
                        type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
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


"""
def init_logger(log_dir: str):
    if os.path.exists(log_dir):
        logs = []
        for fn in os.listdir(log_dir):
            name, ext = os.path.splitext(fn)
            if ext != '.log':
                continue
            logs.append(int(name))

        log_idx = 1 if not logs else max(logs) + 1
    else:
        os.mkdir(log_dir)

        log_idx = 1
    log_fn = os.path.join(log_dir, f'{log_idx:05}.log')

    logging.basicConfig(filename=log_fn, level=logging.INFO, format='%(asctime)s - %(message)s')
"""


def init_logger():
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def log_data_quality(data_quality: dict[str, typing.Any]):
    na_by_col = data_quality['na_by_col'].tolist()
    rows_with_na = data_quality['rows_with_na']
    logging.info(f'na_by_col: {na_by_col}')
    logging.info(f'rows_with_na: {100 * rows_with_na}%')


# Initialize data transformer
data_transformer = DataTransformer(TIMESTAMPS, na_method='median-mode', ctg_method='ohe')
# Initialize ML model
model = RandomForestClassifier()
# Initialize parameters grid
params = {'n_estimators': [1, 2, 4],
          'max_depth': [4, 16, 64, 256],
          }
# Initialize ModelPipeline
pipeline = ModelPipeline(data_transformer, model, params, PATH_TO_MODEL_PIPELINE_SAVES)


@profile
def train(args: argparse.Namespace):
    assert args.data is not None
    #assert args.logs is not None
    assert args.n_iter is not None

    # Initialize logger
    #init_logger(args.logs)
    init_logger()

    # Initialize data provider
    raw_data_path = args.data
    data_provider = DataProvider(raw_data_path, TIMESTAMPS, PATH_TO_DATA_PROVIDER_SAVES)

    # Initialize data analyzer
    data_analyzer = DataAnalyzer()

    if args.verbose:
        print('Training starts')
    print(f'Current position in data: {data_provider.i}')
    time_start = datetime.now()
    i = 0
    etalon_stat = None
    while True:
        # Receive data batch
        data = data_provider.get_batch()
        if data.empty:
            break
        if args.verbose:
            print('Receive new data')
        logging.info(f'Get {data.shape[0]} samples')

        # Analyze data
        stat = data_analyzer.analyze(data)
        log_data_quality(stat['na'])
        if etalon_stat is not None:
            a = np.array(stat['num_means'])
            b = np.array(etalon_stat['num_means'])
            #cos = a @ b / (a ** 2).sum() ** 0.5 / (b ** 2).sum() ** 0.5
            cos_dist = cosine(a, b)
            if cos_dist > 0.2:
                print('Data drift detected')
                etalon_stat = stat
        else:
            etalon_stat = stat

        x, y = data_to_xy(data, TARGET)
        # Evaluate model
        if pipeline.is_fit():
            logging.info('Evaluate model')
            if args.verbose:
                print('Evaluate model')
            score = pipeline.eval(x, y)
            print(f'Score: {score}')
            logging.info(f'Score: {score}')
        # Train model
        if args.verbose:
            print('Train model')
        pipeline.fit(x, y)

        i += 1
        if i == args.n_iter:
            break

        # Emulate delay between data arrivals
        time.sleep(PAUSE)
    time_end = datetime.now()
    print(f'Training time: {time_end - time_start}')
    logging.info(f'Training time: {time_end - time_start}')
    if args.verbose:
        print('Training ends')


@profile
def update(args: argparse.Namespace):
    assert args.data is not None
    #assert args.logs is not None

    # Initialize logger
    #init_logger(args.logs)
    init_logger()

    # Initialize data provider
    raw_data_path = args.data
    data_provider = DataProvider(raw_data_path, TIMESTAMPS, PATH_TO_DATA_PROVIDER_SAVES)

    # Initialize data analyzer
    data_analyzer = DataAnalyzer()

    if not pipeline.is_fit():
        print('Model is not fitted')
        return

    print(f'Current position in data: {data_provider.i}')
    # Receive data batch
    data = data_provider.get_batch()
    if not data.empty:
        if args.verbose:
            print('Receive new data')
        logging.info(f'Get {data.shape[0]} samples')

        # Analyze data
        stat = data_analyzer.analyze(data)
        log_data_quality(stat['na'])

        x, y = data_to_xy(data, TARGET)
        # Train model
        if args.verbose:
            print('Train model')
        pipeline.refit(x, y)


@profile
def evaluate(args: argparse.Namespace):
    assert args.data is not None
    #assert args.logs is not None
    assert args.n_iter is not None

    # Initialize logger
    #init_logger(args.logs)
    init_logger()

    # Initialize data provider
    raw_data_path = args.data
    data_provider = DataProvider(raw_data_path, TIMESTAMPS, PATH_TO_DATA_PROVIDER_SAVES)

    # Initialize data analyzer
    data_analyzer = DataAnalyzer()

    if args.verbose:
        print('Evaluation starts')
    print(f'Current position in data: {data_provider.i}')
    scores = []
    i = 0
    while True:
        # Receive data batch
        data = data_provider.get_batch()
        if data.empty:
            break
        if args.verbose:
            print('Receive new data')
        logging.info(f'Get {data.shape[0]} samples')

        # Analyze data
        stat = data_analyzer.analyze(data)
        log_data_quality(stat['na'])

        x, y = data_to_xy(data, TARGET)
        # Evaluate model
        if pipeline.is_fit():
            logging.info('Evaluate model')
            if args.verbose:
                print('Evaluate model')
            score = pipeline.eval(x, y)
            scores.append(score)
            print(f'Score: {score}')
            logging.info(f'Score: {score}')

        # Emulate delay between data arrivals
        time.sleep(PAUSE)
        i += 1
        if i == args.n_iter:
            break
    print(f'Mean score: {sum(scores) / len(scores)}')
    if args.verbose:
        print('Evaluation ends')


@profile
def inference(args: argparse.Namespace):
    assert args.data is not None
    assert args.out is not None

    # Load data
    x = pd.read_csv(args.data)

    if args.verbose:
        print('Inference starts')
    if pipeline.is_fit():
        # Predict
        x, y = pipeline.predict(x)
        # Save predictions
        xy = xy_to_data(x, y)
        xy.to_csv(args.out)
    else:
        print('Model is not fitted')
    if args.verbose:
        print('Inference ends')


def main():
    # Get args from console
    args = get_args()
    # Read args from config
    read_config(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'update':
        update(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'summary':
        pass
    else:
        print('No mode is specified')


main()
