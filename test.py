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
  	            [[1,2,3], [4,5,6]], 
  	            [[7,8,9], [10, 11, 12]]
  	            ])

X = np.array([[[1,4,7,10],
	          [2,5,8,11],
	          [3,6,9,12]],
	         [[13,16,19,22],
	          [14,17,20,23],
	          [15,18,21,24]]])

print(X.shape) # 2 3 by 4s

U = np.array([[1,3,5], [2,4,6]])





sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  # print(a.get_shape())
  # print(unfold_tf(a,2).eval().shape)
  print(a.eval()) # numpy.ndarray
  x = unfold_tf(a,2).eval()
  y = unfold_np(a.eval(),2)
  # print(x.dot(x.T).shape)
  # print(top_components(x.dot(x.T), 2, 0))
  print(n_mode_prod(X, U, 1).eval())

  # assuming xy is the mode-1 unfolding of a 
  # 3d tensor, we fold it back to 2,3,3 
  # print(refold_tf(xy, [3,3,2], 0).eval())



