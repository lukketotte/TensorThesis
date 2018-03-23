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
from scipy import stats

import logging
logging.basicConfig(filename = 'time_consumption.log', level = logging.DEBUG)
_log = logging.getLogger('time')

# for storing results
dataset = "1_uniform"

a = 0.5
b = 1.5
tucker_rank = [12]*3
dim_pc = 20
pc_rank = 25
compression = float(tucker_rank[0])/float(dim_pc)
max_rank = pc_rank + 5
number_of_runs = 100
# store results
result_runs_core = [None] * number_of_runs
time_runs_core = [None] * number_of_runs
result_runs_x = [None] * number_of_runs
time_runs_x = [None] * number_of_runs
condition_number = []

for i in range(number_of_runs):
	# plot the first result that is seeded
	if i == 0:
		# condition number goes up with higher valued interval
		np.random.seed(1234)
		A = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
		B = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
		C = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
	else:
		A = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
		B = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
		C = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)

	# condition numbers of A, B, C
	_, sA, _ = np.linalg.svd(A)
	_, sB, _ = np.linalg.svd(B)
	_, sC, _ = np.linalg.svd(C)
	condition_number.append(np.amax(sA)/np.amin(sA))
	condition_number.append(np.amax(sB)/np.amin(sB))
	condition_number.append(np.amax(sC)/np.amin(sC))

	X = kruskal_to_tensor([A,B,C])
	X_tl = tl.tensor(X)
	
	core, tucker_factors = tucker(X_tl, ranks = tucker_rank,
		init = "random", tol = 10e-5, random_state = 1234, 
		n_iter_max = 100, verbose = False)

	core_np = tl.to_numpy(core)
	# estimate errors, take the time aswell
	start_time = time.time()
	result_runs_x[i] = error_parafac(tensor = X, max_rank = max_rank, init = "hosvd", verbose = False)
	time_runs_x[i] = time.time() - start_time

	start_time = time.time()
	result_runs_core[i] = error_parafac(tensor = core_np, max_rank = max_rank, init = "hosvd", verbose = False)
	time_runs_core[i] = time.time() - start_time
	print("Completed run %d \n" % (i+1))

# log the relevant results
_log.debug('%d, %.3f, %.3f, %.3f, %s' %(tucker_rank[0], 
								  stats.describe(time_runs_x)[2], 
								  stats.describe(time_runs_core)[2],
								  stats.describe(condition_number)[2],
								  dataset))

xint = range(0, max_rank + 1, 5)
# plot from the other runs aswell with transparancy to get a
# idea of the stability between runs from non seeded results
plt.figure(figsize=(7.5,5))
for i in range(number_of_runs):
	if i == 0:
		continue
	else:
		plt.plot(result_runs_core[i], alpha = 0.01, color = "black")
		plt.plot(result_runs_x[i], alpha = 0.01, color = "black")
# seeded results
x = plt.plot(result_runs_x[0], color = "blue" ,linestyle = '--', label = "Original tensor")
g = plt.plot(result_runs_core[0], color = "red", linestyle = "-", label = "Core tensor")
rank = plt.axvline(x = pc_rank, color = "black", linestyle = ":", label = "Rank($\mathcal{X}$)")
# labels for axes
plt.ylabel("Training Error", fontsize = 16)
plt.xlabel("Tensor Rank", fontsize = 16)
# plt.title("%.2f compression" % (1-compression))
plt.title("$\mathcal{G} \in \Re^{%dx%dx%d}$" % (tucker_rank[0],tucker_rank[1],tucker_rank[2]), 
	fontsize = 24)
plt.legend(loc = "upper right", fontsize = 18)
plt.grid(True)
plt.xticks(xint, fontsize = 14)
plt.yticks(fontsize = 12)
# plt.yticks(np.arange(min(X_error), max(X_error) + 0.05, 0.1), fontsize = 14)
plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\%s_%d" % (dataset, tucker_rank[0]))
# plt.show()


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