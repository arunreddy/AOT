import numpy as np


class Dataset(object):
  def __init__(self, name, X, Y, L, D, YZERO, FZERO):
    self.params = {}
    self.m = 0
    self.n = 0
    self.name = name
    self.X = X
    self.L = L
    self.D = D
    self.Y = Y
    self.YZERO = YZERO
    self.FZERO = FZERO

  def split_train_test(self, m, random_seed=0):
    pos_idx, = np.where(self.Y[:, 0] == 1)
    neg_idx, = np.where(self.Y[:, 0] == -1)

    # Shuffle the index.
    np.random.seed(random_seed)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # Split into positive and negative
    tr_pos_idx = pos_idx[:m]
    te_pos_idx = pos_idx[m:]
    tr_neg_idx = neg_idx[:m]
    te_neg_idx = neg_idx[m:]

    # Split into training and test data.
    tr_idx = np.hstack((tr_pos_idx, tr_neg_idx))
    te_idx = np.hstack((te_pos_idx, te_neg_idx))
    all_idx = np.hstack((tr_idx, te_idx))

    # Rearrange the variables.
    XL = self.X[tr_idx, :]
    YL = self.Y[tr_idx, :]
    XU = self.X[te_idx, :]
    YU = self.Y[te_idx, :]

    L = self.L
    L = L[all_idx, :]
    L = L[:, all_idx]

    D = self.D[all_idx]

    YZERO_L = self.YZERO[tr_idx, :]
    YZERO_U = self.YZERO[te_idx, :]

    FZERO_L = self.FZERO[tr_idx, :]
    FZERO_U = self.FZERO[te_idx, :]

    return XL, YL, XU, YU, L, D, YZERO_L, YZERO_U, FZERO_L, FZERO_U
