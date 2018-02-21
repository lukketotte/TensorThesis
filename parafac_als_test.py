from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from parafac_als import parafac as pf
from tucker_als import TuckerDecomposition as td
from numpy import random as random

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

X = random.normal(loc = 0, scale = 1, size = (30,30,30)) 

pf_test = pf(init = "random", row_info = "original")
pf_test.X_data = X
pf_test.rank = 4
pf_test.init_factors()
# pf_test.row_info("original")
pf_test.parafac_ALS()

x_est = pf_test.reconstruct_X_data()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	# print(x_est.eval())
	x_est_np = x_est.eval()





	