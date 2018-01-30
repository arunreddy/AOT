"""
Read the data from database and generate tf-idf/word2vec features.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import logging
from aot.utils.log_utils import f_timeit

class FeaturesGenerator(object):
  def __init__(self):
    self.logger = logging.getLogger(__name__)

  def tf_idf(self, D):
    vectorizer = TfidfVectorizer(ngram_range=[1, 4], min_df=20, max_df=3000)
    return vectorizer.fit_transform(D)

  def tf_bin(self, D):
    vectorizer = CountVectorizer(binary=True, ngram_range=[1, 4], min_df=20, max_df=3000)
    return vectorizer.fit_transform(D)

  def genereate_features(self, df, t='tf-idf', txt_column_name='review_txt'):

    if t == 'tf-idf':
      X = self.tf_idf(df[txt_column_name])

    elif t == 'bin':
      X = self.tf_bin(df[txt_column_name])

    # elif t == 'word2vec':
    #   obj = Doc2VecUtil(df[txt_column_name])
    #   X = obj.fit()

    return X

  def graph_laplacian(self, X, t_sim='cosine', normed=True):

    # Compute the similarity matrix A
    if t_sim == 'cosine':
      A = cosine_similarity(X)
      A[A < .1] = 0.

    elif t_sim == 'rbf':
      A = rbf_kernel(X, gamma=1.)
      self.logger.debug(np.max(A), np.mean(A), np.min(A))
      A[A < .2] = 0.

    # Laplacian.
    L, D = csgraph.laplacian(A, normed=normed, return_diag=True)
    L = sparse.csc_matrix(L)
    L.eliminate_zeros()
    # self.logger.debug('Nonzeros:', L.nnz)
    # self.logger.debug('Sparsity:', (L.nnz * 100.) / (L.shape[0] * L.shape[1]))

    return L, D

  def generate_features(self, df_pos, df_neg, n, feat_type, sim='cosine'):

    self.logger.debug("> Generating features..")
    self.logger.debug("\t Positive: %d" % (df_pos.shape[0]))
    self.logger.debug("\t Negative: %d" % (df_neg.shape[0]))
    # Combine all the data.
    df = df_pos.copy().append(df_neg)

    # Laplacian matrix.
    X = self.genereate_features(df, feat_type)
    L, D = self.graph_laplacian(X, t_sim=sim)

    # Known labels.
    y = np.asarray([1] * df_pos.shape[0] + [-1] * df_neg.shape[0])

    # OTSC labels.
    y_prime = np.hstack((df_pos['stanford_label'].values, df_neg['stanford_label'].values))
    y_prime = y_prime - 2

    # OTSC confidence scores.
    f_prime = np.hstack((df_pos['f_prime'].values, df_neg['f_prime'].values))

    # Reshape the arrays.
    f_prime = f_prime.reshape(f_prime.shape[0], 1)
    y_prime = y_prime.reshape(y_prime.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    return X, L, D, y, y_prime, f_prime, list(df['review_txt'])

  def generate_features_for_image(self, X, y, X_A, y_A, sim):

    X = TfidfTransformer().fit_transform(X)

    L, D = self.graph_laplacian(X, t_sim=sim)

    y = y * 2
    y = y - 1

    y_A = y_A * 2
    y_A = y_A - 1

    # Compute the scores from auxilliary data.
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()

    X_A = TfidfTransformer().fit_transform(X_A)

    L, D = self.graph_laplacian(X, t_sim=sim)
    clf.fit(X_A, y_A)

    y_prime = clf.predict(X)
    f_prime = np.max(clf.predict_proba(X), axis=1)

    from sklearn.metrics import accuracy_score
    self.logger.debug(accuracy_score(y, y_prime))

    f_prime = np.multiply(y_prime, f_prime)

    return X, L, D, y, y_prime, f_prime
