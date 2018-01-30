from aot.algo.base_algorithm import BaseAlgorithm
import numpy as np

class AOT(BaseAlgorithm):

  def __init__(self):
    self.K = 100
    self.m = 200


  def execute_algorithm(self, dataset, mu_1=0.8, sparse_constraint=False):

    XL, YL, XU, YU, L, D, YZERO_L, YZERO_U, FZERO_L, FZERO_U = dataset.split_train_test(m=self.m)

    lipschitz_constant = np.trace(L + np.identity(L.shape[0])*mu_1)

    # Algorithm
    print("AOT: ", end='')
    for i in range(self.K):

      # Algorithm-1


      # Algorithm-2





      # Metrics


      print("=", end='')

    print("=>[DONE]")


