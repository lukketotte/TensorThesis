from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math
import time


tucker_rank = 15
dim_pc = 20
pc_rank = 20
compression = float(tucker_rank)/float(dim_pc)
max_rank = 25

### Generate factor matricies
# parameters
mean = [0]*pc_rank
# multicollinearity in covA, on avarage 50% of the entries
covA  = covariance_matrix(dim = pc_rank, diagonal = False, seed = 91)
covB = covariance_matrix(dim = pc_rank, diagonal = False, seed = 9110)
covC = covariance_matrix(dim = pc_rank, diagonal = False,  seed = 911021)

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
X_error = error_parafac(tensor = X, max_rank = max_rank, init = "random", verbose = False)
x_time = time.time() - start_time

start_time = time.time()
core_error = error_parafac(tensor = core_np, max_rank = max_rank, init = "random", verbose = False)
core_time = time.time() - start_time

print(x_time, core_time)

xint = range(0, max_rank + 1, 2)
plt.plot(X_error)
plt.plot(core_error)
plt.axvline(x = pc_rank, color = "gray", linestyle = "dashed")
plt.ylabel("Training Error")
plt.xlabel("Tensor Rank")
plt.title("$\mathcal{X} \in \Re^{20x20x20}$, %.2f compression" % compression)
plt.legend(["Original data", "Core tensor"], loc = "upper right")
plt.grid(True)
plt.xticks(xint)
plt.show()
