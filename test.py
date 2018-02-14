from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tucker import TuckerDecomposition as td

X = np.array([[[1,4,7,10],
	          [2,5,8,11],
	          [3,6,9,12]],
	         [[13,16,19,22],
	          [14,17,20,23],
	          [15,18,21,24]]])

print(shuffled([0,1,2,3,4,5,6]))


U1 = np.array([[0.2, 0.4], [0.2,0.3], [0.5, 0.2]])
U2 = np.array([[.1,.3,.5], [.2,.4,.6]])
U3 = np.array([[.2, .4, .9, .2],
	           [.3, .2, .1, .8],
	           [.2, .6, .7, .1]])
 

# X = tf.constant(X, dtype = "float64")
U1 = tf.constant(U1)
U2 = tf.constant(U2)
U3 = tf.constant(U3)

matList = [U1, U2, U3]
print(len(matList))

# X1 = unfold_tf(X, 1)



test = td(X_data = X, shape = [2,3,4], ranks = [2,2,2])
test.hosvd(X)
"""
sess = tf.Session()
with sess.as_default():
  assert tf.get_default_session() is sess
  print(X1.eval())
  print(top_components(X.eval(), 3, 1))
  test = td(X_data = X.eval(), shape = [2,3,4], ranks = [2,2,2])
  
  test.hosvd(X.eval())
"""