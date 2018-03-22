import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math
import time

# for storing results
dataset = "1.1_uniform"

a = 0.5
b = 1.5
tucker_rank = 16
dim_pc = 20
pc_rank = 25
compression = float(tucker_rank)/float(dim_pc)
max_rank = pc_rank + 5

# condition number goes up with higher valued interval
np.random.seed(1234)
A = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
B = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
C = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)

X = kruskal_to_tensor([A,B,C])
X_tl = tl.tensor(X)

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

print(x_time, core_time)

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


"""
X = np.random.uniform(low = 1, high = 2, size = 20).reshape(5,4)

u,s,v = np.linalg.svd(X, full_matrices = False)

w,v = np.linalg.eig(np.dot(X.T, X))

print((w[0]/w[3])**0.5)

s[0] += 2
X = np.dot(u, np.dot(np.diag(s), v))
#print(X)
w,v = np.linalg.eig(np.dot(X.T, X))
print((w[0]/w[3])**0.5)
"""