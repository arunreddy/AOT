from aot.algo.aot import AOT

def execute_algorithm(dataset, algo):

  print(algo)

  if algo == "aot":
    algo = AOT()
    algo.execute_algorithm(dataset)

  elif algo == "AOT-1":
    pass
  elif algo == "AOT-2":
    pass
  else:
    pass



  return None