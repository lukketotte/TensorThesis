# testing the classes
from utils import unfold_tf
from utils import unfold_np
from utils import top_components
from utils import khatri_rao
from utils import refold_tf
from utils import n_mode_prod

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
  	            [[1,2,3], 
  	            [4,5,6]], 
  	            [[7,8,9], 
  	            [10, 11, 12]]
  	            ])

X = np.array([[[1,4,7,10],
	          [2,5,8,11],
	          [3,6,9,12]],
	         [[13,16,19,22],
	          [14,17,20,23],
	          [15,18,21,24]]])

print(X.shape) # 2 3 by 4s

U = np.array([[1,3,5], [2,4,6]])

X = tf.constant(X)
U = tf.constant(U)


sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  # print(a.get_shape())
  # print(x.dot(x.T).shape)
  # print(top_components(x.dot(x.T), 2, 0))
  print(X.eval())
  print(X.get_shape())
  print(unfold_tf(X, 1).eval())
  print(n_mode_prod(X, U, 1).eval())

