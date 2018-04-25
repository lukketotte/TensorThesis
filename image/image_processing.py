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
from math import ceil
import time
from scipy import stats
import matplotlib.image as mpimg

#################

img = mpimg.imread("todd.jpg")
img = np.array(img, dtype = np.float64)
print(img.shape)

compr = 0.9
tucker_rank = [round(compr * img.shape[0]),
			   round(compr * img.shape[1]),
			   img.shape[2]]


img_tl = tl.tensor(img)

core, tucker_factors = tucker(img_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_img = tl.to_numpy(core)

print(core_img.shape)

max_rank = 50

# estimate errors, take the time aswell
start_time = time.time()
X_error = error_parafac(tensor = img, max_rank = max_rank, init = "random", verbose = False)
x_time = time.time() - start_time
print("Time original data: " + str(x_time))

start_time = time.time()
core_error = error_parafac(tensor = core_img, max_rank = max_rank, init = "random", verbose = False)
core_time = time.time() - start_time
print("Time core, 10%: " + str(core_time))


compr = 0.8

tucker_rank = [round(compr * img.shape[0]), 
	           round(compr * img.shape[1]),
	           round(compr * img.shape[2])]

core, tucker_factors = tucker(img_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_img = tl.to_numpy(core)

start_time = time.time()
core_error_2 = error_parafac(tensor = core_img, max_rank = max_rank, init = "random", verbose = False)
core_time = time.time() - start_time
print("Time core, 20%: " + str(core_time))

xint = range(0, max_rank + 1, 5)
# plt.figure(figsize=(9,5))
plt.figure(figsize=(8,5))
plt.plot(X_error, color = "blue" ,linestyle = '--')
plt.plot(core_error, color = "red", linestyle = "-")
plt.plot(core_error_2, color = "green", linestyle = "-.")
#plt.plot(diff_1, color = "blue" ,linestyle = '-')
#plt.plot(diff_2, color = "red", linestyle = '--')

plt.ylabel("Training Error", fontsize = 16)
plt.xlabel("Tensor Rank", fontsize = 16)
# plt.title("%.2f compression" % (1-compression))
# plt.title("$\mathcal{G} \in \Re^{%dx%dx%d}$" % (tucker_rank[0],tucker_rank[1],tucker_rank[2]), 
# 	fontsize = 24)
plt.title("", 
 	fontsize = 24)
plt.legend(["original data","10% compression", "20% compression"], loc = "upper right",
	fontsize = 16)
plt.grid(True)
plt.xticks(xint, fontsize = 14)
plt.yticks(fontsize = 12)
# plt.yticks(np.arange(min(X_error), max(X_error) + 0.05, 0.1), fontsize = 14)
plt.savefig(fname = "C:\\Users\\rotmos\\Dropbox\\Master Thesis\\Thesis\\Figures\\results\\Real\\todd_diff")
# plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Real\\big_all_digits_1")
