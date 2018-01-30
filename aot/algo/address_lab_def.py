from aot.algo.base_algorithm import BaseAlgorithm


class AddressLabelDeficiency(BaseAlgorithm):
  def __init__(self):
    self.K = 100
    self.m = 200

  def execute_algorithm(self, dataset, sparse_constraint=False):
    XL, YL, XU, YU, L, D = dataset.split_train_test(m=self.m)

    # Algorithm
    print("AddressLabelDeficiency: ", end='')
    for i in range(self.K):





      # Metrics


      print("=", end='')

    print("=>[DONE]")

