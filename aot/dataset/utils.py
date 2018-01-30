from aot.dataset.imdb import Imdb


def get_dataset(name, n=100):
  """
  Return the requested dataset.

  :param name: name of the dataset.
  :return: return the instance to dataset.
  """
  dataset = None
  if name == "imdb":
    imdb = Imdb()
    dataset = imdb.get_dataset(n)

  return dataset


def get_DB():
  DB_ENGINE = 'postgresql://postgres@localhost:5432/otsc'
  return DB_ENGINE
