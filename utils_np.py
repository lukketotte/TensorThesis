import numpy as np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd
import scipy.linalg
import scipy.sparse.linalg

def top_components(X, rank, n):

	X_ = unfold_np(X, n)
	A = X_.dot(X_.T)
	# A is square symmetric
	N = A.shape[0]
	# take the N-rank to N-1 eigenvalues
	_, U = eigh(A, eigvals = (N - rank, N - 1))
	# ::-1 returns the reversed order
	U = np.array(U[: , ::-1])
	return U

def hosvd_init(X, n, r):
	"""
	hosvd initation
	X: data tensor
	n: mode, for A^n
	r: desired number of columns

	returns U[:, :r] and boolean depending on r is 
	to large or not
	"""
	xn = unfold_np(X, n)
	shape = list(xn.shape)
	u, _,_ = svd(xn)

	if(r > shape[0]):
		r_to_large = True
		return u, r_to_large
	else:
		r_to_large = False
		return u[:, :r], r_to_large

def unfold_np(arr, ax):
	"""
	unfold np array along its ax-axis
	from: https://gist.github.com/nirum/79d8e14da106c77c02c1
	"""
	return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def refold_np(unfolded_tensor, mode, shape):
	"""
	refold ndarray that was unfolded along the mode into
	ndarray of shape
	"""
	full_shape = list(shape)
	mode_dim = full_shape.pop(mode)
	full_shape.insert(0, mode_dim)
	return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def norm(tensor, order = 2, axis = None):
	
	if axis == ():
		axis = None
	if order == 'inf':
		return np.max(np.abs(tensor), axis = axis)
	elif order == 1:
		return np.sum(np.abs(tensor), axis = axis)
	elif order == 2:
		return np.sqrt(np.sum(tensor ** 2, axis = axis))
	else:
		return np.sum(np.abs(tensor) ** order, axis = axis)**(1/order)


def get_fit(X, Y):
	"""
	Compute squared frobenius norm:
	||X - Y||_F^2  = <X,X> + <Y,Y> - 2 <X,Y>
	run within a tf.Session() so we have numpy arrays
	"""
	########################
	normX = (X ** 2).sum()
	normY = (Y ** 2).sum()
	norm_inner = np.multiply(X,Y).sum()

	norm_residual = normX + normY - 2*norm_inner
	# return 1 - (norm_residual / normX)
	return norm_residual

def khatri_rao(A,B):
	"""
	following the outline of Kolda & Bader (2009),
	column-wise kronecker-product.
	A & B: two matricies with the same number of columns
	"""
	if(A.shape[1] == B.shape[1]):
		rank = A.shape[1]
		rows = A.shape[0] * B.shape[0]
		# khatri row product of A and B will be rows x rank matrix
		P = np.zeros((rows,rank))
		for i in range(rank):
			# kronecker product of two vectors is the outer product
			ab = np.outer(A[:, i], B[:, i])
			# ab is vectorized before assigned to the i'th column of P
			P[:,i] = ab.flatten()
		return P
	else: 
		raise ValueError("Matricies must have the same # of columns")

def khatri_rao_list(A, skip_mat = None, rev = False):
	"""
	Takes a list of factor matricies and constructs the khatri rao
	of all matricies in list.

	If rev = True reverses order of matricies in A

	"""
	N = len(A)
	# rank = list(A[0].shape)[1]
	return_matrix = None

	if not isinstance(skip_mat, type(None)):
		A = A[:skip_mat] + A[(skip_mat + 1):]

	if rev is True:
		A = A[::-1]

	for n in range(N-1):
		# first step
		if isinstance(return_matrix, type(None)):
			return_matrix = khatri_rao(A[n], A[n+1])
		else:
			return_matrix = khatri_rao(return_matrix, A[n+1])

	return return_matrix


def mode_prod(X, A, n):
	"""
	Calculates the n-mode product of tensor X and matrix A
	"""
	shape_a = list(A.shape)
	shape_x = list(X.shape)

	if shape_x[n] == shape_a[1]:
		xn = unfold_np(X, n)
		xn = np.dot(np.transpose(xn), np.transpose(A))
		# folded tensor will have nth dim shape_a[0]
		shape_x[n] = shape_a[0]
		return refold_np(xn, n, shape_x)
	else:
		raise ValueError("X{} ({}) and A{} ({},{}) not defined".format(n, shape[n], n,shape_A[0], shape_A[1]))

def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
	"""n-mode product of a tensor and several matrices or vectors
	Parameters
	----------
	tensor : ndarray
	matrix_or_vec_list : list of matrices or vectors of lengh ``tensor.ndim``
	skip : None or int, optional, default is None
		if not None, index of a matrix to skip
		Note that in any case, `modes`, if provided, should have a lengh of ``tensor.ndim``
	modes : None or int list, optional, default is None
	transpose : bool, optional, default is False
				if True, the matrices or vectors in in the list are transposed
	Returns
	-------
	ndarray, tensor times each matrix or vector in the list at mode `mode`
	Notes
	-----
	If no modes are specified, just assumes there is one matrix or vector per mode and returns:
	:math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`
	"""
	if modes is None:
		modes = range(len(matrix_or_vec_list))

	decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

	res = tensor

	for i, (matrix_or_vec, mode) in enumerate(zip(matrix_or_vec_list, modes)):
		if (skip is not None) and (i == skip):
			continue

		if transpose:
			res = mode_prod(res, np.transpose(matrix_or_vec), mode - decrement)
		else:
			res = mode_prod(res, matrix_or_vec, mode - decrement)

		if np.ndim(matrix_or_vec) == 1:
			decrement = 1

	return res

def kruskal_to_tensor(factors, weights=None):

	shape = [np.shape(factor)[0] for factor in factors]
	if weights is not None:
		full_tensor = np.dot(factors[0]*weights, np.transpose(khatri_rao(factors[1:])))
	else:
		full_tensor = np.dot(factors[0], np.transpose(khatri_rao_list(factors[1:])))
	return refold_np(full_tensor, 0, shape)

def kr(matrices):
	n_columns = matrices[0].shape[1]
	n_factors = len(matrices)

	start = ord('a')
	common_dim = 'z'
	target = ''.join(chr(start + i) for i in range(n_factors))
	source = ','.join(i+common_dim for i in target)
	operation = source+'->'+target+common_dim
	return np.einsum(operation, *matrices).reshape((-1, n_columns))

def khatri_rao_tl(matrices, skip_matrix=None, reverse=False):
	if skip_matrix is not None:
		matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

	# Khatri-rao of only one matrix: just return that matrix
	if len(matrices) == 1:
		return matrices[0]

	n_columns = matrices[0].shape[1]

	# Optional part, testing whether the matrices have the proper size
	for i, matrix in enumerate(matrices):
		if np.ndim(matrix) != 2:
			raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, T.ndim(matrix)))
		if matrix.shape[1] != n_columns:
			raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                                 i, matrix.shape[1], n_columns))

	n_factors = len(matrices)

	if reverse:
		matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices even outside this function

	return kr(matrices)

def partial_svd(matrix, n_eigenvecs=None):
	"""Computes a fast partial SVD on `matrix`
	if `n_eigenvecs` is specified, sparse eigendecomposition
	is used on either matrix.dot(matrix.T) or matrix.T.dot(matrix)
	
	Parameters
	----------
	matrix : 2D-array
	n_eigenvecs : int, optional, default is None
		if specified, number of eigen[vectors-values] to return
	Returns
	-------
	U : 2D-array
		of shape (matrix.shape[0], n_eigenvecs)
		contains the right singular vectors
	S : 1D-array
		of shape (n_eigenvecs, )
		contains the singular values of `matrix`
	V : 2D-array
		of shape (n_eigenvecs, matrix.shape[1])
		contains the left singular vectors
	"""
	# Check that matrix is... a matrix!
	if matrix.ndim != 2:
		raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
			matrix.ndim))

	# Choose what to do depending on the params
	dim_1, dim_2 = matrix.shape
	if dim_1 <= dim_2:
		min_dim = dim_1
	else:
		min_dim = dim_2

	if n_eigenvecs is None or n_eigenvecs >= min_dim:
		# Default on standard SVD
		U, S, V = scipy.linalg.svd(matrix)
		U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
		return U, S, V

	else:
		# We can perform a partial SVD
		# First choose whether to use X * X.T or X.T *X
		if dim_1 < dim_2:
			S, U = scipy.sparse.linalg.eigsh(np.dot(matrix, matrix.T), k=n_eigenvecs, which='LM')
			S = np.sqrt(S)
			V = np.dot(matrix.T, U * 1/S[None, :])
		else:
			S, V = scipy.sparse.linalg.eigsh(np.dot(matrix.T, matrix), k=n_eigenvecs, which='LM')
			S = np.sqrt(S)
			U = np.dot(matrix, V) * 1/S[None, :]

		# WARNING: here, V is still the transpose of what it should be
		U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
		return U, S, V.T