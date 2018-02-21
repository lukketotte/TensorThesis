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


pf_test = pf(init = "hosvd", row_info = "original")
pf_test.X_data = X
pf_test.rank = 2
pf_test.init_factors()
# pf_test.row_info("original")
pf_test.parafac_ALS()

x_est = pf_test.reconstruct_X_data()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

init_op = tf.global_variables_initializer()
with sess:
	sess.run(init_op)
	# print(x_est.eval())
	x_est_np = x_est.eval()

print(x_est_np)
print("\n")
norm_est = (x_est_np ** 2).sum()
print(norm_est)
norm_x = (X ** 2).sum()
print(norm_x)
norm_inner = np.multiply(x_est_np, X).sum()
print(norm_inner)

print(norm_x + norm_est - 2*norm_inner)



	