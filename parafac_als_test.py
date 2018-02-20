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
# st.get_factor_matricies()

# tf.slice(input_, begin, size, name=None)
#x = tf.slice(X, [2,0,0], [2,2,3])
# tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print(U[0].get_shape())
	print(U[1].get_shape())
	print(U[2].get_shape())
	#print(x.eval())

	print(U[1].eval())
	