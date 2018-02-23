from utils_np import *
import numpy as utils_np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd

from parafac_np import parafac
from tucker_np import tucker

import matplotlib.pyplot as plt

X = np.array([[[1.,4.,7.,10.],
	          [1.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

# X = np.random.normal(2, 8, 10*10*10).reshape(10,10,10)

tk = tucker()
tk.X_data = X
tk.ranks = [2,2,2]
tk.init_components()
G = tk.partial_tucker()
A = tk.get_component_mats()

#multi_mode_dot(core, factors, skip=skip_factor, transpose=transpose_factors)
X_tk_est = multi_mode_dot(G, A, None, False)
print(X_tk_est)


pc = parafac(init = "hosvd")


pc.X_data = X
pc.rank = 4

pc.init_factors()
error = pc.parafac_als()
X_hat = pc.reconstruct_X()

# print(error)
"""
plt.plot(error)
plt.ylabel('Training error')
plt.xlabel('Epochs')
plt.title("N(2, 8), 10 x 10 x 10")
plt.grid(True)
plt.show()
"""
