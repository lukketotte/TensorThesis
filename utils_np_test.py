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

Y = np.array([[[1., 2., 3.],
			   [4. ,5. ,6.]],
			  [[7. ,8., 9.],
			   [10., 11., 12.]]])

# X = np.random.normal(loc = 0, scale = 1, size = 20).reshape(2,5,2)

pc = parafac(init = "hosvd")


pc.X_data = X
pc.rank = 6

pc.init_factors()
pc.parafac_als()
X_hat = pc.reconstruct_X()
print(X_hat)
print(get_fit(X, X_hat))

"""

for e in range(self.epochs):
					for mode in range(self._order):
						pseudo_inverse = np.ones([self._rank, self._rank])

						for i, A in enumerate(self.A):
							if i != mode:
								pseudo_inverse[:] = pseudo_inverse * np.dot(np.transpose(A), A)
						pseudo_inverse = normalize(pseudo_inverse)
						A = np.dot(unfold_np(self._X_data, mode), khatri_rao_list(self.A[:mode] + self.A[(mode + 1):]))
						A = np.transpose(np.linalg.solve(np.transpose(pseudo_inverse), np.transpose(A)))
						if not mode == self._order: 
							A = normalize(A)
						self.A[mode] = A

for e in trange(self.epochs):
					for n in range(self._order):
						# V hadamard product of all A(i)'A(i) but nth
						# just brute force for now
						n_list = self.A[:n] + self.A[(n+1): ]
						V = 1
						for i in range(self._order - 1):
							V = np.multiply(V,np.dot(np.transpose(n_list[i]), n_list[i]))
						khatri_list = khatri_rao_list(n_list)
						self.A[n] = np.dot(unfold_np(self._X_data, n), np.dot(khatri_list, pinv(V)))
						
					rec_error = norm(self._X_data - self.reconstruct_X(), 2) / norm_X
					rec_errors.append(rec_error)
"""
