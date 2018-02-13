from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh

X = np.array([[[1,4,7,10],
	          [2,5,8,11],
	          [3,6,9,12]],
	         [[13,16,19,22],
	          [14,17,20,23],
	          [15,18,21,24]]])


U1 = np.array([[0.2, 0.4], [0.2,0.3], [0.5, 0.2]])
U2 = np.array([[.1,.3,.5], [.2,.4,.6]])
U3 = np.array([[.2, .4, .9, .2],
	           [.3, .2, .1, .8],
	           [.2, .6, .7, .1]])
 

X = tf.constant(X, dtype = "float64")
U1 = tf.constant(U1)
U2 = tf.constant(U2)
U3 = tf.constant(U3)

matList = [U1, U2, U3]
print(len(matList))

sess = tf.Session()

with sess.as_default():
  assert tf.get_default_session() is sess
  # print(a.get_shape())
  # print(x.dot(x.T).shape)
  # print(top_components(x.dot(x.T), 2, 0))
  #print(X.eval())
  print(matList[0].get_shape())
  print(X.get_shape())
  print(unfold_tf(X, 2).get_shape())
  #print(n_mode_prod(X, U2, 1).eval())
  print(kruskal(X, matList).eval())

