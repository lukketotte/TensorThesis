from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from parafac_als import parafac as pf

X = np.array([[[2.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])


pf_test = pf(init = "random")
pf_test.X_data = X
pf_test.rank = 3
pf_test.init_factors()


# xn = unfold_tf(pf_test.X_data, 1) # 3 by 8
# slice to get column 1
# xns = tf.slice(xn, begin = [0, 0], size = [3,1])



pf_test.parafac_ALS()

U = pf_test.get_factor_matricies()

test1 = kruskal_tf_parafac(U)
test2 = kruskal_tf_parafac([U[2], U[1], U[0]])

x_est = pf_test.reconstruct_X_data()
xn = unfold_tf(pf_test.X_data, 0)
# slice first two columns
# xn_2 = tf.slice(xn, begin = [0,0], size = [3,2])

norm = tf.norm(U[0], axis = 1)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print(U[0].eval())
	print(U[1].eval())
	print(U[2].eval())
	print(x_est.eval())
	# print(norm.eval())
	# print(test1.eval())
	# print(test2.eval())


	




	