from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tucker import TuckerDecomposition as td
from tucker_test import TuckerDecomposition as tdt
# from tensorly import decomposition as dp
# import tensorly as tl

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

xt = tf.get_variable("xt", dtype = tf.float64, 
	initializer = X)




# X1 = unfold_tf(X, 1)

# test = tdt(shape = [2,3,4], ranks = [2,2,2], X_data = X)
# X, G, A = test.hooi()
# test = td(X_data = X, shape = [2,3,4], ranks = [2,2,2], epochs = 200)
# hosvd = test.hosvd(X)
# print(hosvd)
# decompX = test.hooi(X)
# print(decompX)
# X_est, G, A = test.hooi()
y = tf.matmul(unfold_tf(xt, 2), tf.transpose(unfold_tf(xt, 2)))
v = tf.svd(y, compute_uv = True)[2]
v = v[:, :2]
init_op = tf.global_variables_initializer()
kron = tf.contrib.kfac.utils.kronecker_product(y, v)

with tf.Session() as sess:
	sess.run(init_op)
	# print(xt.eval())
	# print(y.eval())

	# print(s.eval())
	print("\n")
	print(v.eval())
	print("\n")
	print(v.eval().shape)
	print(y.eval().shape)
	print(kron.eval().shape)

	

