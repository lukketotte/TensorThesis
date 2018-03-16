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

mean = [0,0,0]

cov = [[1.5, 1.7, 8.1],
	   [1.7, 3, 0.7],
	   [8.1, 0.7, 2.2]]

tucker_rank = 5
dim_pc = 10
compression = float(tucker_rank)/float(dim_pc)
max_rank = 14

np.random.seed(1234)
A = np.random.multivariate_normal(mean, cov, dim_pc*10).reshape(3,10,10)
# 10x10x10 tensor
X = kruskal_to_tensor(A)
X_tl = tl.tensor(X)

# Tucker decomposiion
tucker_rank = [tucker_rank]*3
core, tucker_factors = tucker(X_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_np = tl.to_numpy(core)

X_error = error_parafac(tensor = X, max_rank = max_rank,
	init = "hosvd", verbose = True)
core_error = error_parafac(tensor = core_np, max_rank = max_rank,
	init = "hosvd", verbose = True)

xint = range(0, max_rank + 1, 2)
plt.plot(X_error)
plt.plot(core_error)
plt.ylabel("Training Error")
plt.xlabel("Tensor Rank")
plt.title("$\mathcal{X} \in \Re^{10x10x10}$, %.2f compression" % compression)
plt.legend(["Original data", "Core tensor"], loc = "upper right")
plt.grid(True)
plt.xticks(xint)
plt.show()


