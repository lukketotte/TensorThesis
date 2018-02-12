# testing the classes
from utils import unfold_tf
from utils import unfold_np
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh


# 3-D tensor `a`
# [[[ 1,  2,  3],
#   [ 4,  5,  6]],
#  [[ 7,  8,  9],
#   [10, 11, 12]]]
# a = tf.constant(np.arange(1, 13, dtype=np.int32),
#                 shape=[2, 2, 3])
a = tf.constant([
  	            [[1,2,3], [4,5,6]], 
  	            [[7,8,9], [10, 11, 12]]
  	            ])

sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  # print(a.get_shape())
  # print(unfold_tf(a,2).eval().shape)
  print(type(a.eval())) # numpy.ndarray
  x = unfold_tf(a,2).eval()
  y = unfold_np(a.eval(),2)

  print(x.dot(x.T).shape)




