from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math
import time

import logging
logging.basicConfig(filename = 'time_consumption.log', level = logging.DEBUG)
_log = logging.getLogger('time')

# for storing results
dataset = "21c"

tucker_rank = 18
dim_pc = 20
pc_rank = 25
compression = float(tucker_rank)/float(dim_pc)
max_rank = pc_rank + 5

### Generate factor matricies
# parameters
mean = [0]*pc_rank
# multicollinearity in covA, on avarage 50% of the entries
covA  = covariance_matrix(dim = pc_rank, diagonal = False, seed = 1234)
covB = covariance_matrix(dim = pc_rank, diagonal = False, seed = 1234)
covC = covariance_matrix(dim = pc_rank, diagonal = False,  seed = 1234)

# generate factor matricies
A = np.random.multivariate_normal(mean, covA, size = pc_rank)
B = np.random.multivariate_normal(mean, covB, size = pc_rank)
C = np.random.multivariate_normal(mean, covC, size = pc_rank)

# Testing the other simulations scheme
# covariance_matrix_parafac(10, 5, [1,1,1])
factor_matricies =  covariance_matrix_parafac(dim_pc, pc_rank, [1,1,1])

# A = np.random.multivariate_normal(mean, cov, dim_pc*pc_rank).reshape(3,dim_pc,pc_rank)
# 10x10x10 tensor

# X = kruskal_to_tensor([A,B,C])
X = kruskal_to_tensor(factor_matricies)
X_tl = tl.tensor(X)



# Tucker decomposiion
tucker_rank = [tucker_rank]*3
core, tucker_factors = tucker(X_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_np = tl.to_numpy(core)

# estimate errors, take the time aswell
start_time = time.time()
X_error = error_parafac(tensor = X, max_rank = max_rank, init = "hosvd", verbose = False)
x_time = time.time() - start_time

start_time = time.time()
core_error = error_parafac(tensor = core_np, max_rank = max_rank, init = "hosvd", verbose = False)
core_time = time.time() - start_time



_log.debug('%d, %.2f, %.3f, %.3f, %s' %(pc_rank, (1-compression), x_time, core_time, dataset))

xint = range(0, max_rank + 1, 5)

plt.plot(X_error, color = "black" ,linestyle = '--')
plt.plot(core_error, color = "black", linestyle = "-")
plt.axvline(x = pc_rank, color = "black", linestyle = ":")
plt.ylabel("Training Error", fontsize = 16)
plt.xlabel("Tensor Rank", fontsize = 16)
# plt.title("%.2f compression" % (1-compression))
plt.title("$\mathcal{G} \in \Re^{%dx%dx%d}$" % (tucker_rank[0],tucker_rank[1],tucker_rank[2]), 
	fontsize = 24)
plt.legend(["Original data", "Core tensor", "rank($\mathcal{X}$)"], loc = "upper right",
	fontsize = 18)
plt.grid(True)
plt.xticks(xint, fontsize = 14)
plt.yticks(fontsize = 12)
# plt.yticks(np.arange(min(X_error), max(X_error) + 0.05, 0.1), fontsize = 14)
# plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\%s_%d" % (dataset, tucker_rank[0]))
plt.show()
