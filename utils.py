import numpy as np
import tensorflow as tf
from scipy.linalg import eigh

"""
Helper methods for the main classes used in the other
classes.
https://github.com/ebigelow/tf-decompose/blob/master/utils.py
"""

def top_components(X, rank, n):
  """
  Based on algorithm 1 in 
  T. Kolda, 2006, Multilinear Operators for Higher-Order Decompositions, Sandia
  Formula is A^(n) <- J_n leading eigenvalues of X_n * X_n'

  Important note: any tf.tensor returned by .eval() is a numpy.ndarray
  """
  X_ = unfold_np(X, n)
  A = X_.dot(X_.T)
  N = A.shape[0]

def unfold_np(arr, ax):
  """
  unfold np array along its ax-axis
  from: https://gist.github.com/nirum/79d8e14da106c77c02c1
  """
  return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def unfold_tf(X, n):
  """
  unfold TF variable X along the n-axis
  input shape: (I_1, ..., I_N)
  output shape: (d_n, I_1 * ... * I_n-1 * I_n+1 * ... * I_N)
  """
  shape = X.get_shape().as_list()
  # subtract one from each element
  idxs = [i for i,_ in enumerate(shape)]

  new_idxs = [n] + idxs[:n] + idxs[(n+1):]
  B = tf.transpose(X, new_idxs)

  dim = shape[n]

  return tf.reshape(X, [dim, -1])

def refold_tf(X, shape, n):
  """
  X: tf variable
  shape: desired shape of output tensor
  n: assume X is unrolled along shape[n]
  """
  # subtracts one from each element in shape list
  idxs = [i for i,_ in enumerate(shape)]

  shape_temp = [shape[n]] + shape[:n] + shape[(n+1):]
  B = tf.reshape(X, shape_temp)
  new_idxs = idxs[1:(n+1)] + [0] + idxs[(n+1):]
  return tf.transpose(B, new_idxs)

"""
Functions to add
# Outer product as in the PARAFAC type model
# mode-n multiplication
# 
"""