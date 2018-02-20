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

U = pf_test.get_factor_matricies()

# xn = unfold_tf(pf_test.X_data, 1) # 3 by 8
# slice to get column 1
# xns = tf.slice(xn, begin = [0, 0], size = [3,1])


krusk = kruskal_tf(U[0], U[1], 3)
# should be 6 by 3
all_krusk = kruskal_tf_parafac(U)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print(U[0].eval())
	print(U[1].eval())

	print(U[0].get_shape())
	print(U[1].get_shape())
	print(U[2].get_shape())
	print("\n")
	print(krusk.eval())
	print(krusk.get_shape())
	y = np.kron(U[0].eval()[:, 0], U[1].eval()[: , 0])
	# final result for column 0 of khatri rao prod of all 
	# 3 U matricies
	print(np.kron(y, U[2].eval()[:,0]))

	print(all_krusk.eval())



	