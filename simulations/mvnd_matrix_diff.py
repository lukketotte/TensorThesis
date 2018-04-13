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

# for storing results
dataset = "23_mvnd"
diag = False
a = 1
b = 2
tucker_rank = [16]*3
dim_pc = 20
pc_rank = 15
mean = [0]*pc_rank
compression = float(tucker_rank[0])/float(dim_pc)
max_rank = pc_rank + 5
number_of_runs = 100
# store results
result_runs_core = [None] * number_of_runs
time_runs_core = [None] * number_of_runs
result_runs_x = [None] * number_of_runs
time_runs_x = [None] * number_of_runs
diff_result = [None] * number_of_runs
diff_mean = 0

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

	temp_g = np.array(result_runs_core[i])
	temp_x = np.array(result_runs_x[i])
	diff_result[i] = temp_x - temp_g
	diff_mean += (temp_x - temp_g)/number_of_runs

	print("Completed run %d \n" % (i+1))

# should perhaps rather use the mean all the iterations
# diff_mean = []
# for i in range(number_of_runs):
#	diff_mean.append(np.mean(diff_result[i]))

xint = range(0, max_rank + 1, 5)
# plot from the other runs aswell with transparancy to get a
# idea of the stability between runs from non seeded results
plt.figure(figsize=(9,5))
for i in range(number_of_runs):
	#if i == 0:
		#continue
	#else:
	plt.plot(diff_result[i], alpha = 0.04, color = "black")
# seeded results
# x = plt.plot(diff_result[0], color = "blue" ,linestyle = '-')
x = plt.plot(diff_mean, color = "blue" ,linestyle = '-')
rank = plt.axvline(x = pc_rank, color = "black", linestyle = ":")
# labels for axes
plt.ylabel("Difference in Training Error", fontsize = 16)
plt.xlabel("Tensor Rank", fontsize = 16)
# plt.title("%.2f compression" % (1-compression))
plt.title("{0}% compression".format(int(100*round(1 - tucker_rank[0]/dim_pc, 1))), 
	fontsize = 24)
plt.grid(True)
plt.xticks(xint, fontsize = 14)
plt.yticks(fontsize = 12)
# plt.yticks(np.arange(min(X_error), max(X_error) + 0.05, 0.1), fontsize = 14)
plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\Mvnd\\%s_%d_diff" % (dataset, tucker_rank[0]))
# plt.savefig(fname = "C:\\Users\\rotmos\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Simulations\\%s_%d" % (dataset, tucker_rank[0]))
plt.show()
