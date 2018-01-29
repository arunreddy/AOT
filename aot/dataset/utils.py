from aot.dataset.imdb import Imdb


def get_dataset(name, n):
  """
  Return the requested dataset.

  :param name: name of the dataset.
  :return: return the instance to dataset.
  """

  if name == "imdb":
    imdb = Imdb()
    df_pos, df_neg = imdb.load_data_from_raw_files()
    return df_pos, df_neg


  return None


def get_DB():
  DB_ENGINE = 'postgresql://postgres@localhost:5432/otsc'
  return DB_ENGINE