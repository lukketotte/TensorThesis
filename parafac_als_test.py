from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from parafac_als import parafac as pf
from tucker_als import TuckerDecomposition as td

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])


pf_test = pf(init = "random", row_info = "original")
pf_test.X_data = X
pf_test.rank = 6
pf_test.init_factors()
# pf_test.row_info("original")
pf_test.parafac_ALS()


x_est = pf_test.reconstruct_X_data()

tucker = td()
tucker.rank = [2,2,2]
tucker.X_data = X
G = tucker.tucker_ALS()
# get core tensor
G = refold_tf(G, [2,2,2], 0)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	# print(x_est.eval())
	G_np = G.eval()

tf.reset_default_graph()
pf_core = pf(init = "random", row_info = "core")
pf_core.X_data = G_np
pf_core.rank = 6
pf_core.init_factors()
# pf_test.row_info("core")
pf_core.parafac_ALS()
	




	