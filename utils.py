import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from scipy.linalg import kron

"""
Helper methods for the main classes used in the other
classes.
https://github.com/ebigelow/tf-decompose/blob/master/utils.py
"""

def top_components(X, rank, n):
	"""
	Based on algorithm 1 in 
	T. Kolda, 2006, Multilinear Operators for Higher-Order Decompositions, Sandia
	Formula is A^(n) <- J_n leading eigenvalues of X_n * X_n'
	Not sure if correctly interpreted, but for now it returns the 
	eigenvectors corresponding to the highest eigenvalues

	Important note: any tf.tensor returned by .eval() is a numpy.ndarray
	"""
	# for a 2d tensor the unfolding leaves the matrix intact so no
	# need to do any input validation in terms of shape
	X_ = unfold_np(X, n)
	A = X_.dot(X_.T)
	# A is square symmetric
	N = A.shape[0]
	# take the N-rank to N-1 eigenvalues
	_, U = eigh(A, eigvals = (N - rank, N - 1))
	# ::-1 returns the reversed order
	U = np.array(U[: , ::-1])
	return U

def shuffled(ls):
	return sorted(list(ls), key=lambda _: np.random.rand())

def unfold_np(arr, ax):
	"""
	unfold np array along its ax-axis
	from: https://gist.github.com/nirum/79d8e14da106c77c02c1
	"""
	return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def unfold_tf(X, n):
	"""
	unfold TF variable X along the n-axis
	input shape: (I_1, ..., I_N)
	output shape: (d_n, I_1 * ... * I_n-1 * I_n+1 * ... * I_N)
	"""
	shape = X.get_shape().as_list()
	# subtract one from each element
	idxs = [i for i,_ in enumerate(shape)]

	new_idxs = [n] + idxs[:n] + idxs[(n+1):]
	B = tf.transpose(X, new_idxs)

	dim = shape[n]

	return tf.reshape(B, [dim, -1])

def refold_tf(X, shape, n):
	"""
	X: tf variable
	shape: desired shape of output tensor
		   for 3d [matricies, rows, cols]
	n: assume X is unrolled along shape[n]
	"""
	# subtracts one from each element in shape list
	idxs = [i for i,_ in enumerate(shape)]

	shape_temp = [shape[n]] + shape[:n] + shape[(n+1):]
	B = tf.reshape(X, shape_temp)
	new_idxs = idxs[1:(n+1)] + [0] + idxs[(n+1):]
	return tf.transpose(B, new_idxs)

def get_fit(X, Y):
	"""
	Compute squared frobenius norm:
	||X - Y||_F^2  = <X,X> + <Y,Y> - 2 <X,Y>
	run within a tf.Session() so we have numpy arrays
	"""
	normX = (X ** 2).sum()
	normY = (Y ** 2).sum()
	norm_inner = (X * Y).sum()

	norm_residual = normX + normY - norm_inner
	# fit as percentage, lower values for closer fit
	return 1 - (norm_residual / normX)

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
  	        # of the two
			ab = np.outer(A[:, i], B[:, i])
			# ab is vectorized before assigned to the i'th column of P
			P[:,i] = ab.flatten()
		return P
	else: 
		raise ValueError("Matricies must have the same # of columns")

def n_mode_prod(X, A, n, X_shape = None):
	"""
	Calculates the n-mode product of a tensor and matrix A
	"""
	shape_A = list(A.get_shape())

	if X_shape is None:
		shape = list(X.get_shape())
	else:
		shape = X_shape
	# check that dimensions allows for matrix multiplication
	# list index out of range, lies in the shape_A thing

	if shape[n] == shape_A[1]:
		Xn = unfold_tf(X, n)
		Xn = tf.matmul(tf.transpose(Xn), A)
		# alternatively
		# Xn = np.matmul(Xn, A.T)

		# the folded tensor will have nth dimension A.shape[0]
		shape[n] = shape_A[0]
		print(Xn)

		return refold_tf(Xn, shape, n)

	else:
		raise ValueError("X{} ({}) and A{} ({},{}) not defined".format(n, shape[n], n,shape_A[0], shape_A[1]))


def update_component_matricies(U, X, n):
	"""
	N.D Sidiopulous et. al, 2016, Tensor Decompositions for
	Signal processing and Machine Learning, IEEE

	orthogonal tucker ALS update helper
	U1 = r1 prinp of (V kron W)' *  X1' 
	returns (V kronw W)' * X1'


	Note: V kron W = W kron V

	U: list of component matricies
	X: data tensor
	n: axis

	TODO: how would the update step look like for 
		  the n-dimensional tensor?

	"""
	Xn = unfold_tf(X, n)
	# will keep this hardcoded for 3d case for now
	# for now:
	if len(U) == 3:
		# remove the n'th matrix from list of 
		# component mataricies
		U = U[:n] + U[(n+1):]
		A = tf.contrib.kfac.utils.kronecker_product(U[0], U[1])
		# take the tranpose of Xn as the paper unfolds in that way
		# in comparison with Kolda, which the unfolding function
		# was writting in accordance with
		A = tf.matmul(tf.transpose(A), tf.transpose(Xn))

		return(A)
	else:
		raise ValueError("Currently only allowing for 3d tensors")

def mpinv(A, reltol = 1e-10):
	"""
	Computes the moore-penrose inverse, clearing any 
	values less than reltol
	"""
	s, u, v = tf.svd(A)

	# invert s, clear entries lower than reltol*s[0]
	atol = tf.reduce_max(s) * reltol
	s = tf.boolean_mask(s, s > atol)
	s_inv = tf.diag(tf.concat([1. / s], 0))

	# compute v * s_inv * u_t:
	return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))

def kruskal_np(G, U, X_shape = None):
	"""
	Kruskal operator, takes core matrix and list och 
	unfolded matricies.

	G: tensor 
	U: list of factor matricies

	X = G times1 U1 times2 U2 ... timesN UN
	"""
	# need to keep in mind that the shape returns
	# the dimension in a completely different order in 
	# comparison with the litterature. 
	N = len(U)
	for n in range(N):
		# n_mode_prod does the unfolding and refolding
		if X_shape is None:
			X = n_mode_prod(G, U[n], n)
		else:
			X = n_mode_prod(G, U[n], n, X_shape)
	return(X)

def kruskal_tf(A, B, r):
	"""
	helper method for kruskal_tf_parafac()
	kronecker prod of two vectors = vectorized outer product
	# TODO: return has an ineffecient transpose, revisit and 
			check if fixable
	"""
	Ia = A.get_shape()[0]
	Ib = B.get_shape()[0]
	col_list = [None] * r
	for n in range(r):
		a = tf.slice(A, begin = [0, n], size = [Ia, 1])
		b = tf.slice(B, begin = [0, n], size = [Ib, 1])
		col = tf.matmul(a, tf.transpose(b))
		col_list[n] = tf.reshape(col, [-1])

	return tf.transpose(tf.concat([col_list], 1))
	

def kruskal_tf_parafac(A):
	"""
	Kruskal product of list of tf.tensors. The i'th column
	will be the kronecker products of the i'th column vectors of 
	all tensors in A. 

	This method is specifically coded for the parafac solution
	assuming A is RxR
	"""
	N = len(A)
	r = A[0].get_shape()[1]
	temp = None

	for n in range(N-1):
		# first step
		if isinstance(temp, type(None)):
			temp = kruskal_tf(A[n], A[n+1], r)
		else:
			temp = kruskal_tf(temp, A[n+1], r)

	return temp

	









