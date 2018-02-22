import numpy as np
from utils import *
import tensorly as tl
from tensorly import decomposition
from utils_np import *
from tensorly import tenalg
from tensorly_parafac_algo import parafac

X = np.array([[[1.,4.,7.,10.],
	          [2.,5.,8.,11.],
	          [3.,6.,9.,12.]],
	         [[13.,16.,19.,22.],
	          [14.,17.,20.,23.],
	          [15.,18.,21.,24.]]])

a = np.array([[1.5,2.2,3.3],
			  [4.4,5.54,6.3],
			  [7.7,8.2,9.34]])

b = np.array([[1.5,2.23,5.3],
			  [2.4,5.54,6.3],
			  [7.7,6.2,2.34]])

c = np.array([[1.2,2.21,5.3],
			  [2.4,5.54,1.3],
			  [2.7,6.2,2.24]])

# print(tenalg.khatri_rao([a,b,c], skip_matrix = 2))
# print(khatri_rao_list([a,b]))


X0 = unfold_np(X,0)
X = refold_np(X0, 0, (2,3,4))


print(norm(X))
factors_new = parafac(X, rank = 6)

Xt = tl.tensor(X)

factors = decomposition.parafac(Xt, rank = 3, init = "random")

khatri = khatri_rao_list(factors_new[1:])
x0 = np.dot(factors_new[0], np.transpose(khatri))
X_new = refold_np(x0, 0, [2,3,4])

# reconstruct
X_est = tl.kruskal_to_tensor(factors)
print(X_est)
print(X_new)
print("\n")
print(get_fit(Xt, X_est))
print(get_fit(X, X_new))



