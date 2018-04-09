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
logging.basicConfig(filename = 'congrunce.log', level = logging.DEBUG)
_log = logging.getLogger('time')

# for storing results
dataset = "1_uniform"

a = .5
b = 1.5
tucker_rank = [18,18,20]
dim_pc = 20
pc_rank = 25
compression = float(tucker_rank[0])/float(dim_pc)
max_rank = pc_rank + 5
number_of_runs = 5
# store results
result_runs_core = [None] * number_of_runs
time_runs_core = [None] * number_of_runs
result_runs_x = [None] * number_of_runs
time_runs_x = [None] * number_of_runs
condition_number = []

# for congruence estimates
congruence_A = []
congruence_B = []
congruence_C = []
congruence_A_tucker = []
congruence_B_tucker = []
congruence_C_tucker = []

np.random.seed(1234)
A = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
B = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)
C = np.random.uniform(low = a, high = b, size = dim_pc*pc_rank).reshape(dim_pc,pc_rank)

X = kruskal_to_tensor([A,B,C])

"""
# split half
congrunce =  split_half_analysis(X, 2, congruence_rank, 
	split_type = "half")
# print("Congrunce for run %d" % (i+1))
# print(congrunce)
congruence_A.append(congrunce[0])
congruence_B.append(congrunce[1])
congruence_C.append(congrunce[2])
"""

X_tl = tl.tensor(X)
	
core, tucker_factors = tucker(X_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_np = tl.to_numpy(core)

# number of runs
runs = 100
# rank for congruence analysis
congruence_rank = 3
for i in range(runs):
	idxs = np.random.choice(20, 20, replace = False)
	# original data using permutation
	congrunce =  split_half_analysis(X, 2, congruence_rank, 
	split_type = idxs)
	congruence_A.append(congrunce[0])
	congruence_B.append(congrunce[1])

	# core 
	congrunce_tucker = split_half_analysis(core_np, 2, congruence_rank, 
	split_type = idxs)
	congruence_A_tucker.append(congrunce_tucker[0])
	congruence_B_tucker.append(congrunce_tucker[1])

print("X")
print(stats.describe(congruence_A)[2],
      stats.describe(congruence_A)[3]**.5)
print(stats.describe(congruence_B)[2],
	  stats.describe(congruence_B)[3]**.5)
print("G")
print(stats.describe(congruence_A_tucker)[2],
      stats.describe(congruence_A_tucker)[3]**.5)
print(stats.describe(congruence_B_tucker)[2],
	  stats.describe(congruence_B_tucker)[3]**.5)
