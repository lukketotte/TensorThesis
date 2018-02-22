import numpy as np
from scipy.linalg import eigh
from scipy.linalg import kron
from numpy.linalg import svd

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
	"""
	Computes l-'order' norm of tensor

	Params
	------
	tensor: ndarray
	order: int
	axis: int or tuple

	Returns
	------
	float or tensor
		if 'axis' is provided returns a tensor
	"""
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

def khatri_rao_list(A, skip_mat = None, rev = True):
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

	if shape[n] == shape_a[1]:
		xn = unfold_np(X, n)
		xn = np.dot(np.transpose(xn), A)
		# folded tensor will have nth dim shape_a[0]
		shape_x[n] = shape_a[0]
		return refold_np(xn, n, shape_x)
	else:
		raise ValueError("X{} ({}) and A{} ({},{}) not defined".format(n, shape[n], n,shape_A[0], shape_A[1]))
