from decompositions.parafac_np import parafac
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math
from utils.utils_np import *

"""
Should try to simulate data nowing the ground truth

Assuming its a 3d tensor

TODO: create a class that runs this simulation scheme
"""
np.random.seed(1234)
# X = Kruskal(A,B,C)
# let X = 20,20,20
# rank(X) leq 20^2
# A, B, C: 20 x 10
A = np.random.normal(0, 1, 30*20).reshape(30, 20)
B = np.random.normal(1, 1, 30*20).reshape(30, 20)
C = np.random.normal(2, 1, 30*20).reshape(30, 20)

X = kruskal_to_tensor([A,B,C])
print(X[:,:,1])

Xt = tl.tensor(X)

# Seems like the parafac error behaves exactly
# the same way tucker rank is at least the same 
# as the rank in the data generating process

core, factors = tucker(Xt , ranks = [10,10,10])
tucker_reconstruction = tl.tucker_to_tensor(core, factors)
core = tl.to_numpy(core)

niter = 20
error_iter = [None] * niter
error_iter_g = [None] * niter

pc_x = parafac(X, init = "hosvd")
pc_x.X_data = X

pc_g = parafac(core, init = "hosvd")
pc_g.X_data = core


for i in range(niter):
	pc_x.rank = i
	pc_x.init_factors()
	error_x = pc_x.parafac_als()
	error_iter[i] = error_x[len(error_x) - 1]

	pc_g.rank = i
	pc_g.init_factors()
	error_g = pc_g.parafac_als()
	error_iter_g[i] = error_g[len(error_g) - 1]


print(pc_g.X_data.shape)
print(pc_x.X_data.shape)

xint = range(0, niter + 1, 5)
plt.plot(error_iter)
plt.plot(error_iter_g)
plt.ylabel('Training error')
plt.xlabel('$Rank_{CP}$')
plt.title("$CP_{True} = 20, \,\mathcal{X} \in \Re^{30x30x30}$")
plt.legend(['Original data', 'Core tensor(.33)'], loc='upper right')
plt.grid(True)
plt.xticks(xint)
plt.show()

