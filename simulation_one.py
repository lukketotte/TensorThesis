# This is a first simulation test
# creating a tensor of size 30 x 30 x 5 from N(0,1)

from utils import *
import numpy as np
from numpy import random as random
import tensorflow as tf
from scipy.linalg import eigh
from parafac_als import parafac as pf
from tucker_als import TuckerDecomposition as td

# generate X from N(0,1) of size 20, 20, 20 (in python terms)
X = random.normal(loc = 0, scale = 3, size = (50, 50 ,50))

tucker = td()
tucker.rank = [25,25,25]
tucker.X_data = X
G = tucker.tucker_ALS()
# get core tensor, all dimensions reduced to 50% rounded up
G = refold_tf(G, [25,25,25], 0)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	G_np = G.eval()

max_R = 10
for i in range(max_R - 1):
	pf_test = pf(init = "random", row_info = "original")
	pf_test.X_data = X
	pf_test.rank = i + 1
	pf_test.init_factors()
	# pf_test.row_info("original")
	pf_test.parafac_ALS()

	x_est = pf_test.reconstruct_X_data()

	tf.reset_default_graph()
	
	pf_core = pf(init = "random", row_info = "core")
	pf_core.X_data = G_np
	pf_core.rank = i + 1
	pf_core.init_factors()
	# pf_test.row_info("core")
	pf_core.parafac_ALS()

	tf.reset_default_graph()
