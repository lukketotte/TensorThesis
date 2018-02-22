from utils_np import *
import numpy as utils_np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd
from parafac_np import parafac

X = np.array([[[1.,4.,7.,10.],
	          [1.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

X = np.random.normal(2, 1, 3*4*2).reshape(2,3,4)


Y = np.array([[[1., 2., 3.],
			   [4. ,5. ,6.]],
			  [[7. ,8., 9.],
			   [10., 11., 12.]]])

# X = np.random.normal(loc = 0, scale = 1, size = 20).reshape(2,5,2)

pc = parafac(init = "hosvd")


pc.X_data = X
pc.rank = 1

pc.init_factors()
pc.parafac_als()
X_hat = pc.reconstruct_X()
print(X_hat)
print(get_fit(X, X_hat))

