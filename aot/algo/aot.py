import numpy as np

from aot.algo.address_lab_def import AddressLabelDeficiency
from aot.algo.base_algorithm import BaseAlgorithm


class AOT(BaseAlgorithm):
  def __init__(self):
    self.K = 10
    self.m = 200

  def Q1_Q2(self, L, Y, FZERO, DEL1_F, DEL2_F, mu_1):

    F = FZERO + DEL1_F + DEL2_F
    p1 = F.T*L*F
    p2 = mu_1*np.linalg.norm(F-Y,2)
    p3 = np.sum(np.abs(DEL1_F)) # np.linalg.norm(DEL1_F,'2')

    return p1 + p2 + p3

  def Q1_Q3(self):
    return


  def compute_accuracy(self,FZERO, DEL1_F,DEL2_F,Y,m,n):
    F = FZERO + DEL1_F + DEL2_F
    correct = 0
    for i in range(m,m+n):
      if F[i,0]*Y[i,0] >= 0:
        correct +=1

    return correct/n



  def execute_algorithm(self, dataset, mu_1=0.8, sparse_constraint=False):
    XL, YL, XU, YU, L, D, YZERO_L, YZERO_U, FZERO_L, FZERO_U = dataset.split_train_test(m=self.m)

    m = YL.shape[0]
    n = YU.shape[0]

    Y = np.vstack((YL, YU))
    print(np.unique(Y))
    FZERO = np.vstack((FZERO_L, FZERO_U))
    YZERO = np.vstack((YZERO_L, YZERO_U))
    YZERO = YZERO - 2

    FZERO = np.multiply(FZERO,YZERO)

    DEL1_F = np.zeros((m + n, 1), dtype=float)
    DEL2_F = np.zeros((m + n, 1), dtype=float)

    # Algorithm
    print("AOT: ", end='')
    iteration = 0
    for i in range(self.K):
      # Algorithm-1
      algo1 = AddressLabelDeficiency()
      DEL1_F = algo1.execute_algorithm(L, Y, FZERO, DEL1_F, DEL2_F, mu_1, m, n, maxIterations=10)
      print("\tCost Function: %0.4f" % self.Q1_Q2(L, Y, FZERO, DEL1_F, DEL2_F, mu_1), end=' ')
      print("\tAccuracy: %0.4f"%self.compute_accuracy(FZERO, DEL1_F,DEL2_F,Y,m,n))
      # Algorithm-2


      # Metrics
      iteration += 1
      print("=", end='')

    print("=>[Iterations:%d][DONE]" % iteration)
