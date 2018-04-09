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
dataset = "2_mvnd"
diag = False
a = 0
b = 1
tucker_rank = [18]*3
dim_pc = 20
pc_rank = 25
mean = [0]*pc_rank
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
		covA  = covariance_matrix(dim = pc_rank, diagonal = diag, seed = 1234,
			uniform_params = [a,b])
		covB = covariance_matrix(dim = pc_rank, diagonal = diag, seed = 1234,
			uniform_params = [a,b])
		covC = covariance_matrix(dim = pc_rank, diagonal = diag,  seed = 1234,
			uniform_params = [a,b])
		A = np.random.multivariate_normal(mean, covA, size = pc_rank)
		B = np.random.multivariate_normal(mean, covB, size = pc_rank)
		C = np.random.multivariate_normal(mean, covC, size = pc_rank)
	else:
		covA  = covariance_matrix(dim = pc_rank, diagonal = diag, uniform_params = [a,b])
		covB = covariance_matrix(dim = pc_rank, diagonal = diag, uniform_params = [a,b])
		covC = covariance_matrix(dim = pc_rank, diagonal = diag, uniform_params = [a,b])
		A = np.random.multivariate_normal(mean, covA, size = pc_rank)
		B = np.random.multivariate_normal(mean, covB, size = pc_rank)
		C = np.random.multivariate_normal(mean, covC, size = pc_rank)

	X = kruskal_to_tensor([A,B,C])
	X_tl = tl.tensor(X)

	# print(X[0,0,0])
	
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
_log.debug('%d, %.3f, %.3f, %s' %(tucker_rank[0], 
								  stats.describe(time_runs_x)[2], 
								  stats.describe(time_runs_core)[2],
								  dataset))

xint = range(0, max_rank + 1, 5)
# plot from the other runs aswell with transparancy to get a
# idea of the stability between runs from non seeded results
plt.figure(figsize=(7.5,5))
for i in range(number_of_runs):
	if i == 0:
		continue
	else:
		plt.plot(result_runs_core[i], alpha = 0.025, color = "black")
		plt.plot(result_runs_x[i], alpha = 0.025, color = "black")
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
plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\Mvnd\\%s_%d" % (dataset, tucker_rank[0]))
# plt.savefig(fname = "C:\\Users\\rotmos\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\Mvnd\\%s_%d" % (dataset, tucker_rank[0]))
# plt.show()
