import time

def f_timeit(func):
  def wrapper(*args, **kwargs):
    start_time = time.time()
    print(" --- %s ---"%func.__name__)
    res = func()
    print("--- %0.6f seconds ---" % (time.time() - start_time))
    return res
  return wrapper