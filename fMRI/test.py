import numpy as np

from nilearn import plotting as plot
from nilearn.image import smooth_img
from nilearn import image
from nilearn import datasets
import matplotlib.pyplot as plt

import os as os
import tensorflow as tf
import tensorly as tl

import time

from tensorData import tensor as td
from selectADHD import adhd as adhd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import tensorly as tl
from tensorly.decomposition import tucker

# ----------- ADD DATA ----------- #
# TODO: fetch more data as you get home 
# adhdData = datasets.fetch_adhd(n_subjects = 40)
#func_filenames = adhdData.func
# print(func_filenames[0])

# ---------- INSTANCE of selectADHD ---------- #
# laptop
#csv_loc = "C:\\Users\\lukas\\Documents\\master\\Thesis\\Python"
#csv_loc = os.path.join(csv_loc, "allSubs_testSet_phenotypic_dx.csv")
#nifty_loc = "D:\\Thesis\\Data\\Data\\adhd\\data"

# stationary
csv_loc = "F:\\Thesis\\Data\\Data\\adhd"
csv_loc = os.path.join(csv_loc, "allSubs_testSet_phenotypic_dx.csv")
nifty_loc = "F:\\Thesis\\Data\\Data\\adhd\\data"

# get list of locations for up to 5 subjects for site 1 in folder
# nifty_loc
adhdTest = adhd(csv_loc, nifty_loc, 6, 5)
funcFilenames = adhdTest.listOfLocations()
print(funcFilenames)

# get the data from the locations and append to list img
# this code goes very very slow for some reason

img = []
for i in range(len(funcFilenames)):
 	x = smooth_img(funcFilenames[i], fwhm = "fast").get_data()
	# still not sure about the shapes within sites
 	print(x.shape)
 	img.append(x)


# ------------- INSTANCE OF tensorData ------------- #
# w & y has no data?
# ERROR: every even position has no data
tdTest = td(img, [0,1,2], 3)
a = tdTest.niftyList
# print(len(a))
# this is fine
# print(type(a[0]))
# print(type(a[1]))
# print(a[0][1,1,1,:])
# print(tdTest.idx_spatial)
tensor_cube = tdTest.unfoldTemporal()
print(tensor_cube.shape)
#271633, the dimensions are correct
# print(tensor_cube.shape)
# print(type(tensor_cube))

# works fine
# tfTensor = tf.Variable(tensor_cube)


"""

# ----------- CHECK THE DATA ---------- #
t = np.array(range(0,176))
# print(tensor_cube[:,205000,1])
# plt.plot(t, tensor_cube[:,100000,1])
# plt.show()
# find the first position of non-zeroes
idx = 1
for i in range(100500, 101000):
	if(np.count_nonzero(tensor_cube[:, i, 0]) > 0):
		idx = i
		#print("Found value: col %d" % i)
		break

#a = tensor_cube[:, 100500, 0]
# this is a problem... to much casting. Not sure what is happening
#print(type(a))
#print(type(t))
#plt.plot(t,a)
#plt.show()


# going to test spatialMean function and check time
xt = np.array(range(0,175))

from datetime import datetime
startTime = datetime.now()

# first subject
xt = tdTest.spatialMean(tensor_cube, 0)
print(datetime.now() - startTime)

yt = tdTest.spatialMean(tensor_cube, 1)
# this thing gives an indexing error
# IndexError: index 2 is out of bounds for axis 2 with size 2
wt = tdTest.spatialMean(tensor_cube, 2)
# same here, no data?
# wt = tdTest.spatialMean(tensor_cube, 3)
# print(yt)
plt.plot(t, xt, "r", t, wt,"b", t, yt, "g")
plt.show()

"""

# Analysis, RIP
# start with getting the core tensor
# 20 % compression in all modes but the subject mode
compression = 0.8
tucker_rank = [round(compression * tensor_cube.shape[0]),
			   round(compression * tensor_cube.shape[1]),
			   tensor_cube.shape[2]]

print("Tucker rank: " + str(tucker_rank[0]) + ", " + str(tucker_rank[1]))

tl.set_backend('numpy')
#tensor_cube_tl = tl.tensor(tensor_cube)

# fit the tucker decomposition
core, tucker_factors = tucker(tensor_cube, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 20, verbose = False)

core_cube = tl.to_numpy(core)
print("Core shape: " + str(core_cube.shape))

max_rank = 50

start_time = time.time()
X_error = error_parafac(tensor = tensor_cube, max_rank = max_rank,
	init = "random", verbose = False)
x_time = time.time() - start_time
print("X time: " + str(x_time))

start_time = time.time()
core_error_1 = error_parafac(tensor = core_cube, max_rank = max_rank,
	init = "random", verbose = False)
core_1_time = time.time() - start_time
print("Core 0.2 time: " + str(core_1_time))

###

compression = 0.9
tucker_rank = [round(compression * tensor_cube.shape[0]),
			   round(compression * tensor_cube.shape[1]),
			   tensor_cube.shape[2]]

core, tucker_factors = tucker(tensor_cube_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 250, verbose = False)

core_cube = tl.to_numpy(core)
print("Core shape, 0.1: " + str(core_cube.shape))

start_time = time.time()
core_error_2 = error_parafac(tensor = core_cube, max_rank = max_rank,
	init = "random", verbose = False)
core_2_time = time.time() - start_time
print("Core 0.2 time: " + str(core_2_time))

xint = range(0, max_rank + 1, 5)
# plt.figure(figsize=(9,5))
plt.figure(figsize=(8,5))
plt.plot(X_error, color = "blue" ,linestyle = '--')
plt.plot(core_error_2, color = "red", linestyle = "-")
plt.plot(core_error_1, color = "green", linestyle = "-.")
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
# plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Real\\digit%s" % compression_string)
plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Real\\fmri_tensor")
