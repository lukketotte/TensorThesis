from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tucker_als import TuckerDecomposition as td


# 2x3x4 tensor
X = np.array([[[1.23, 4.5, 2.45, 12.3],
	          [7.4, 5.2, 8.3, 11.9],
	          [3.2, 6.5, 9.2, 14.5]],
	         [[13.1, 12.2, 5.3, 2.3],
	          [8.45, 7.2, 16.3, 4.23],
	          [2.3, 8.56, 1.56, 4.15]]])

Y = np.array([[[2.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])
tucker = td()

# print(tucker.rank)

tucker.rank = [2,3,4]
tucker.X_data = Y

U = tucker.return_component_matricies()


init_op = tf.global_variables_initializer()

xt = tf.get_variable("xt", dtype = tf.float64, 
	initializer = X)

yt = tf.get_variable("yt", dtype = tf.float64, 
	initializer = Y)


G = tucker.tucker_ALS()
G = refold_tf(G, [2,3,4], 0)
U = tucker.return_component_matricies()

X_est = tucker.G_to_X()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	# this works, and op can be defined
	# outside session run stuff
	op = tf.assign(xt, yt)
	sess.run(op)
	print(xt.eval())
	print(G.eval())
	print("\n Shapes comps:")
	print(U[0].get_shape())
	print(U[1].get_shape())
	print(U[2].get_shape())
	print("\n Estimated data tensor: ")
	print(X_est.eval())