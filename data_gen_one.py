from utils import *
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from parafac_als import parafac as pf
from tucker_als import TuckerDecomposition as td
from numpy import random as random

#### GENERATE DATA ####
# generate tensor of true rank 5,
# X is 2 by 5 by 3
A = tf.random_normal(shape = (2,5), mean = 5, dtype = tf.float64)
B = tf.random_normal(shape = (5,5), mean = 0, dtype = tf.float64)
C = tf.random_normal(shape = (3,5), mean = 2, dtype = tf.float64)

X0 = tf.matmul(A, tf.transpose(kruskal_tf_parafac([B,C])))
X = refold_tf(X0, [2,5,3], 0)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	X = X.eval()

########################
print(X.shape)
print(X)
# for i in range(7):
pf_test = pf(init = "hosvd", row_info = "original")
pf_test.rank = 2
pf_test.X_data = X
pf_test.init_factors()
pf_test.parafac_ALS()
tf.reset_default_graph()