from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tucker import TuckerDecomposition as td
from tucker_test import TuckerDecomposition as tdt

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

xt = tf.get_variable("xt", dtype = tf.float64, 
	initializer = X)




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

test = tdt(shape = [2,3,4], ranks = [2,2,2], X_data = X)

# test = td(X_data = X, shape = [2,3,4], ranks = [2,2,2], epochs = 200)
# hosvd = test.hosvd(X)
# print(hosvd)
# decompX = test.hooi(X)
# print(decompX)


"""
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	y = tf.matmul(unfold_tf(xt, 2), tf.transpose(unfold_tf(xt, 2)))
	print(xt.eval())
	print(y.eval())

	u = tf.svd(y, compute_uv = False)
	# u, idx = tf.nn.top_k(u, k = 3, sorted = False)

	print(u.eval())
"""
