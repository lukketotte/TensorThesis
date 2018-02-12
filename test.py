# testing the classes
from utils import unfold_tf
from utils import unfold_np
from utils import top_components
from utils import khatri_rao

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
  print(top_components(x.dot(x.T), 2, 0))

X = np.array([[1,2,3],
			  [4,5,6]])
Y = np.array([[6,5,4],
	          [3,2,1],
	          [2,3,4]])

print(X.shape)
print(Y.shape)
print(khatri_rao(X,Y))


