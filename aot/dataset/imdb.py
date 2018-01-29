"""
Script to covernt the large movie review dataset to numpy format.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


class Imdb(object):
  def __init__(self):
    self.data_dir = os.environ['DATA_DIR'] + "/aclImdb"
    self.db_engine = 'postgresql://postgres@localhost:5432/otsc'

  def read_files(self, dir, name, label):
    reviews = []
    for file in glob(os.path.join(self.data_dir, dir, '*.txt')):
      file_name = os.path.basename(file)
      review_txt = open(file, 'r', encoding='utf8').readlines()[0]
      print('> %s: %d lines' % (file_name, len(review_txt)))
      reviews.append([name, review_txt, label])

    return reviews

  def read_data(self):
    # Read positive reviews

    reviews = []
    reviews.extend(self.read_files('train/pos', 'train', 1))
    reviews.extend(self.read_files('train/neg', 'train', -1))
    reviews.extend(self.read_files('test/pos', 'test', 1))
    reviews.extend(self.read_files('test/neg', 'test', -1))

    df = pd.DataFrame(reviews, columns=['name', 'review_txt', 'label'])
    df['stanford_label'] = 0.0
    df['stanford_confidence_scores'] = ""

    df['nltk_label'] = 0.0
    df['nltk_confidence_scores'] = 0.0

    print(df.head())

    psql = create_engine(self.db_engine)

    df.to_sql('acl_imdb', psql, if_exists='replace')

  def load_data(self, n):
    df_pos = pd.read_sql('SELECT * FROM acl_imdb WHERE label=1 ORDER BY index ASC LIMIT %d ' % n,
                         con=create_engine(self.db_engine), parse_dates=True)
    df_neg = pd.read_sql('SELECT * FROM acl_imdb WHERE label=-1 ORDER BY index ASC LIMIT %d ' % n,
                         con=create_engine(self.db_engine), parse_dates=True)

    df_pos['f_prime'] = df_pos['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
    df_neg['f_prime'] = df_neg['stanford_confidence_scores'].apply(lambda x: np.max(list(map(float, x.split(',')))))
    return df_pos, df_neg

  def load_data_from_raw_files(self, n=100):
    raw_file = os.path.join(os.environ['DATA_DIR'], 'imdb.csv')
    df = pd.read_csv(raw_file)

    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == -1]

    return df_pos.head(n), df_neg.head(n)

  def get_dataset(self, n=100):
    df_pos, df_neg = self.load_data_from_raw_files(n)
    dataset = None

    # Generate features.


    # Split as test and train.


    # Compute the adjacency matrix



    # Create a dataset object


    return dataset


if __name__ == '__main__':
  imdb = Imdb()
  imdb.read_data()
