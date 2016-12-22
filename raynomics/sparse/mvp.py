import ray
import numpy as np
import scipy
import scipy.sparse

ray.register_class(scipy.sparse.csr.csr_matrix)
ray.register_class(scipy.sparse.csc.csc_matrix)

# Code for sparse matrix vector products

def partition_matrix_rows(A, num_partitions):
  n = A.shape[0]
  partition_size = -(-n // num_partitions) # ceil division
  partitions = [A[i:min(i+partition_size, n), :] for i in xrange(0, n, partition_size)]
  return [ray.put(partition) for partition in partitions]

@ray.remote
def dot(A, v):
  "Compute the matrix-vector-product A \cdot v."
  return A.dot(v)

def mvp(A_ids, v):
  """Compute the matrix-vector-product A \cdot v in a distributed fashion."""
  v_id = ray.put(v)
  return np.concatenate(ray.get([dot.remote(rows, v) for rows in A_ids]))
