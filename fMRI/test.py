import numpy as np

from nilearn import plotting as plot
from nilearn.image import smooth_img
from nilearn import image
from nilearn import datasets
import matplotlib.pyplot as plt

import os as os
import tensorflow as tf
import tensorly as tl

from tensorData import tensor as td
from selectADHD import adhd as adhd
# ----------- ADD DATA ----------- #
# TODO: fetch more data as you get home 
adhdData = datasets.fetch_adhd(n_subjects = 40)
func_filenames = adhdData.func
print(func_filenames[0])

# ---------- INSTANCE of selectADHD ---------- #
csv_loc = "C:\\Users\\lukas\\Documents\\master\\Thesis\\Python"
csv_loc = os.path.join(csv_loc, "allSubs_testSet_phenotypic_dx.csv")
nifty_loc = "D:\\MasterThesis\\Data\\adhd\\data"

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
#271633, the dimensions are correct
# print(tensor_cube.shape)
# print(type(tensor_cube))

# works fine
tfTensor = tf.Variable(tensor_cube)


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