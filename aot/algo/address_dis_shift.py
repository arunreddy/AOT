import numpy as np
from aot.algo.base_algorithm import BaseAlgorithm
from sklearn.svm import LinearSVR


class AddressDisShift(BaseAlgorithm):
  def __init__(self):
    self.m = 200


  iteration = 0

  def execute_algorithm(self, L, Y, FZERO, DEL1_F, DEL2_F, mu_1, m, n, maxIterations=10):
    IL = np.vstack((np.ones((m, 1)), np.zeros((n, 1))))
    IL = IL * np.ones(IL.shape[0])

    lipschitz_constant = np.trace(L + IL * mu_1)

    # Algorithm
    print("\tAddressDisShift: ", end='')


    print("=>[Iterations:%d][DONE]" % iteration)
    return DEL2_F
