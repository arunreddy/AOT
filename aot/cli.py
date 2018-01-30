import argparse
import hashlib
import logging
import os
from aot.utils.log_utils import f_timeit
from aot.dataset import  get_dataset
from aot.algo import execute_algorithm

def main(args):

  # The following set of actions are self-explainable.
  dataset = get_dataset(args.dataset, 1000)
  execute_algorithm(dataset, args.algo)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='OTSC - Off-the-Shelf Classifier')
  parser.add_argument('--dataset', choices=['imdb', 'amazon_fine_foods', 'convex', 'cats_dogs'], default='imdb')
  parser.add_argument('--feat', choices=['bin', 'tf-idf', 'word2vec'], default='tf-idf')
  parser.add_argument('--sim', choices=['cosine', 'rbf'], default='cosine')
  parser.add_argument('--n-iterations', type=int, default=10)
  parser.add_argument('--n-total', type=int, default=1000)
  parser.add_argument('--n-labeled', type=int, default=400)
  parser.add_argument('--weighted', action='store_true', default=True)
  parser.add_argument('--debug', action='store_true', default=True)
  parser.add_argument('--algo', choices=['svm', 'baseline', 'aot', 'aot-1', 'aot-2'], default='aot')


  args = parser.parse_args()

  DATA_DIR = '/home/arun/code/github/AOT/data/raw'
  RESULTS_DIR = '/home/arun/code/github/AOT/data/results'
  TMP_DIR = '/home/arun/code/github/AOT/data/tmp'

  os.environ['DATA_DIR'] = DATA_DIR
  os.environ['RESULTS_DIR'] = RESULTS_DIR
  os.environ['TMP_DIR'] = TMP_DIR

  args.data_home = DATA_DIR
  args.results_dir = RESULTS_DIR
  args.tmp_dir = TMP_DIR
  args.args_hash = hashlib.md5(str(args).encode('utf-8')).hexdigest()

  # set up logging.
  log_format = '%(asctime)s - %(name)-12s - %(levelname)-8s: %(message)s'
  log_level = logging.DEBUG if args.debug else logging.INFO
  logging.basicConfig(level=log_level, format=log_format)


  main(args)

