from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math

# going to simulate a tensor where the factor matricies
# are generated from trivariate ND. 


# how should i generate the covariance matrix?


tucker_rank = 8
dim_pc = 20
pc_rank = 10
compression = float(tucker_rank)/float(dim_pc)
max_rank = 12

x = np.random.uniform(0, 1, size = pc_rank).reshape(pc_rank, 1)
y = np.random.uniform(1, 2, size = pc_rank).reshape(pc_rank, 1)
cov = np.dot(x, y.T)
print(cov)
mean = [0]*pc_rank

np.random.seed(1234)
A = np.random.multivariate_normal(mean, np.diag(np.random.rand(pc_rank)), size = pc_rank)
B = np.random.multivariate_normal(mean, np.diag(np.random.rand(pc_rank)), size = pc_rank)
C = np.random.multivariate_normal(mean, np.diag(np.random.rand(pc_rank)), size = pc_rank)

# A = np.random.multivariate_normal(mean, cov, dim_pc*pc_rank).reshape(3,dim_pc,pc_rank)
# 10x10x10 tensor

X = kruskal_to_tensor([A,B,C])
X_tl = tl.tensor(X)



# Tucker decomposiion
tucker_rank = [tucker_rank]*3
core, tucker_factors = tucker(X_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_np = tl.to_numpy(core)

X_error = error_parafac(tensor = X, max_rank = max_rank,
	init = "random", verbose = True)
core_error = error_parafac(tensor = core_np, max_rank = max_rank,
	init = "random", verbose = True)

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
