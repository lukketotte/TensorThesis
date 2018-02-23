from utils_np import *
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

X = np.random.normal(2, 8, 10*10*10).reshape(10,10,10)

Xt = tl.tensor(X)

core, factors = tucker(Xt , ranks = [5,5,5])
tucker_reconstruction = tl.tucker_to_tensor(core, factors)
core = tl.to_numpy(core)


#tk = tucker()
#tk.X_data = X
#tk.ranks = [2,2,2]
#tk.init_components()
#G = tk.partial_tucker()
#A = tk.get_component_mats()



pc = parafac(init = "hosvd")


pc.X_data = X
pc.rank = 2

pc.init_factors()
error_x = pc.parafac_als()
X_hat = pc.reconstruct_X()

pc_g = parafac(init = "hosvd")
pc_g.X_data = core
pc_g.rank = 2
pc_g.init_factors()
error_g = pc_g.parafac_als()

# print(error)

plt.plot(error_x)
plt.plot(error_g)
plt.ylabel('Training error')
plt.xlabel('Epochs')
plt.title("$ N(\mu = 2, \sigma^2 = 8),\mathcal{X} \in \Re^{10x10x10}$")
plt.legend(['Original data', 'Core tensor'], loc='upper right')
plt.grid(True)
plt.show()

