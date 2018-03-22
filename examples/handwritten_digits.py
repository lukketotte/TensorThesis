# set folder to parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from decompositions.parafac_np import parafac
from utils.utils_np import *
from utils.core_parafac_analysis import *

import time

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

print(type(images_and_labels[0]))
"""
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
# plt.show()

image, label = images_and_labels[3]
plt.imshow(image, cmap = plt.cm.gray_r)
plt.axis('off')
plt.title('%d' % label)
# plt.show()
"""

def digit_tensor_maker(img_labels, digit):
	n = len(images_and_labels)
	list_of_digits = []
	for i in range(n):
		image, label = img_labels[i]
		if label == digit:
			list_of_digits.append(image)
	return np.asarray(list_of_digits)

# 183 x 8 x 8 tensor for the digit 3
digit_tensor = digit_tensor_maker(images_and_labels, 0)


# combine all into a 150 x 64 x 10 tensor
digit_tensor = []
all_digits = list(range(0,10))

for idx, digit in enumerate(all_digits):
	temp_tensor = digit_tensor_maker(images_and_labels, digit)[:150, :, :]
	temp_tensor = unfold_np(temp_tensor, 0)
	# each column should correspond to a digit
	digit_tensor.append(temp_tensor.T)

digit_tensor = np.asarray(digit_tensor)





"""
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(digit_tensor[i, :, :], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % digit)
plt.show()
"""

digit_tl = tl.tensor(digit_tensor)




compression = 0.9

# filename
compression_string = str(compression)
compression_string = compression_string.replace("0.", "_")


tucker_rank = [round(compression* digit_tensor.shape[0]), 
	           round(compression * digit_tensor.shape[1]),
	           round(compression * digit_tensor.shape[2])]

core, tucker_factors = tucker(digit_tl, ranks = tucker_rank,
	init = "random", tol = 10e-5, random_state = 1234, 
	n_iter_max = 100, verbose = False)

core_digit = tl.to_numpy(core)

max_rank = 45

# estimate errors, take the time aswell
start_time = time.time()
X_error = error_parafac(tensor = digit_tensor, max_rank = max_rank, init = "hosvd", verbose = False)
x_time = time.time() - start_time

start_time = time.time()
core_error = error_parafac(tensor = core_digit, max_rank = max_rank, init = "hosvd", verbose = False)
core_time = time.time() - start_time
print(x_time, core_time)


xint = range(0, max_rank + 1, 5)

plt.plot(X_error, color = "black" ,linestyle = '--')
plt.plot(core_error, color = "black", linestyle = "-")

plt.ylabel("Training Error", fontsize = 16)
plt.xlabel("Tensor Rank", fontsize = 16)
# plt.title("%.2f compression" % (1-compression))
plt.title("$\mathcal{G} \in \Re^{%dx%dx%d}$" % (tucker_rank[0],tucker_rank[1],tucker_rank[2]), 
	fontsize = 24)
plt.legend(["Original data", "Core tensor"], loc = "upper right",
	fontsize = 18)
plt.grid(True)
plt.xticks(xint, fontsize = 14)
plt.yticks(fontsize = 12)
# plt.yticks(np.arange(min(X_error), max(X_error) + 0.05, 0.1), fontsize = 14)
# plt.savefig(fname = "C:\\Users\\lukas\\Dropbox\\Master Thesis\\Thesis\\Figures\\Results\\Real\\digit_%s" % compression_string)
plt.show()
