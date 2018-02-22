from utils_np import *
import numpy as utils_np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd
from parafac_np import parafac
import matplotlib.pyplot as plt

X = np.array([[[1.,4.,7.,10.],
	          [1.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

X = np.random.normal(2, 8, 10*10*10).reshape(10,10,10)


Y = np.array([[[1., 2., 3.],
			   [4. ,5. ,6.]],
			  [[7. ,8., 9.],
			   [10., 11., 12.]]])

# X = np.random.normal(loc = 0, scale = 1, size = 20).reshape(2,5,2)

pc = parafac(init = "hosvd")


pc.X_data = X
pc.rank = 50

pc.init_factors()
error = pc.parafac_als()
X_hat = pc.reconstruct_X()

# print(error)
plt.plot(error)
plt.ylabel('Training error')
plt.xlabel('Epochs')
plt.title("N(2, 8), 10 x 10 x 10")
plt.grid(True)
plt.show()