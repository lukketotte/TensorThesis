from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tucker_als import TuckerDecomposition as td


# 2x3x4 tensor
X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

Y = np.array([[[2.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])
tucker = td()

# print(tucker.rank)

tucker.rank = [2,2,2]
tucker.X_data = X

U = tucker.return_component_matricies()


init_op = tf.global_variables_initializer()

xt = tf.get_variable("xt", dtype = tf.float64, 
	initializer = X)

yt = tf.get_variable("yt", dtype = tf.float64, 
	initializer = Y)




G = tucker.tucker_ALS()
G = refold_tf(G, [2,2,2], 0)
U = tucker.return_component_matricies()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	# this works, and op can be defined
	# outside session run stuff
	op = tf.assign(xt, yt)
	sess.run(op)
	print(xt.eval())
