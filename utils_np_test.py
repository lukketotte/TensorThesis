from utils_np import *
from core_parafac_analysis import *
import numpy as utils_np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd

from tensorly.decomposition import tucker
import tensorly as tl

from parafac_np import parafac
# from tucker_np import tucker

import matplotlib.pyplot as plt

X = np.array([[[1.,4.,7.,10.],
	          [1.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

X = np.random.normal(2, 8, 40*40*40).reshape(40,40,40)

Xt = tl.tensor(X)

core, factors = tucker(Xt , ranks = [20,20,20])
tucker_reconstruction = tl.tucker_to_tensor(core, factors)
core = tl.to_numpy(core)

error_x = error_parafac(X, 30)
error_g = error_parafac(core, 30)



# print(error)

plt.plot(error_x)
plt.plot(error_g)
plt.ylabel('Training error')
plt.xlabel('Epochs')
plt.title("$ N(\mu = 2, \sigma^2 = 8),\mathcal{X} \in \Re^{10x10x10}$")
plt.legend(['Original data', 'Core tensor'], loc='upper right')
plt.grid(True)
plt.show()

