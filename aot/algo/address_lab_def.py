import numpy as np

from aot.algo.base_algorithm import BaseAlgorithm


class AddressLabelDeficiency(BaseAlgorithm):
  def __init__(self):
    self.m = 200

  def soft_thresh(self, x, l):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - l, 0.))

  def execute_algorithm(self, L, Y, FZERO, DEL1_F, DEL2_F, mu_1, m, n, maxIterations=10):
    IL = np.vstack((np.ones((m, 1)), np.zeros((n, 1))))
    IL = IL * np.ones(IL.shape[0])

    lipschitz_constant = np.trace(L + IL * mu_1)

    # Algorithm
    print("\tAddressLabelDeficiency: ", end='')
    iteration = 0
    t = 1.
    DEL1_F_Y = DEL1_F.copy()

    for i in range(maxIterations):
      DEL1_F_OLD = DEL1_F.copy()

      DEL1_F_Y = DEL1_F_Y - (np.dot(L + IL * mu_1, FZERO + DEL1_F_OLD + DEL2_F) - np.dot(mu_1 * IL,
                                                                                         Y)) / lipschitz_constant
      DEL1_F = self.soft_thresh(DEL1_F_Y, 1. / lipschitz_constant)

      t0 = t
      t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
      DEL1_F_Y = DEL1_F + ((t0 - 1.) / t) * (DEL1_F - DEL1_F_OLD)

      # Metrics
      iteration += 1
      print("=", end='')

    print("=>[Iterations:%d][DONE]" % iteration)
    return DEL1_F
