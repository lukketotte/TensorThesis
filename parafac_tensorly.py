import numpy as np
from utils import *
import tensorly as tl
from tensorly import decomposition

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

X0 = unfold_np(X,0)
print(X0)
X = refold_np(X0, 0, (2,3,4))
print(X)

X = tl.tensor(X)

factors = decomposition.parafac(X, rank = 2)

print(factors[0], factors[1])

# reconstruct
X_est = tl.kruskal_to_tensor(factors)


print(X_est)

print(get_fit(X, X_est))