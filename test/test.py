from __future__ import print_function

import ray
import unittest
import time
import sys

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy
import scipy.sparse

import raynomics.sparse
from raynomics.sparse.mvp import mvp
import raynomics.sparse.irlb as irlb

class SparseMVPTest(unittest.TestCase):

  def testMVP(self):
    # reloading is only neccessary in the test because we call ray.init twice
    reload(raynomics.sparse.mvp)
    ray.init(start_ray_local=True, num_workers=16)

    # Test a matrix-vector-product
    m = 5001
    n = 5003
    A = scipy.sparse.rand(m, n, density=0.001, format="csr")
    v = np.random.rand(n)
    # First compute the single node matrix-vector-product
    w1 = A.dot(v)
    # Now compute the matrix-vector-product with ray and compare
    A_ids = raynomics.sparse.mvp.partition_matrix_rows(A, 8)
    w2 = mvp(A_ids, v)
    assert_almost_equal(w1, w2)
  
    # Here is how to do the transpose
    v = np.random.rand(m)
    At = A.transpose().tocsr()
    w3 = At.dot(v)
    At_ids = raynomics.sparse.mvp.partition_matrix_rows(At, 8)
    w4 = mvp(At_ids, v)
    assert_almost_equal(w3, w4)

    ray.worker.cleanup()

class SparseIRLBTest(unittest.TestCase):

  def testIRLB(self):
    # reloading is only neccessary in the test because we call ray.init twice
    reload(raynomics.sparse.irlb)
    reload(raynomics.sparse.mvp)
    ray.init(start_ray_local=True, num_workers=16)
 
    m = 10000
    n = 5000
    A = scipy.sparse.rand(m, n, density=0.001, format="csr")

    X = irlb.origirlb(A, 50)
    Y = irlb.irlb(A, 50, 8)

    assert_almost_equal(X[1], Y[1])
    assert_almost_equal(X[0], Y[0])

    ray.worker.cleanup()

if __name__ == "__main__":
  unittest.main(verbosity=2)
