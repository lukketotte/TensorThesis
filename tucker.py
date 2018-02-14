import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *
import logging

logging.basicConfig(filename = 'loss.log', level = logging.DEBUG)
_log = logging.getLogger('decomp')


class TuckerDecomposition():
	"""
	Used for Tucker decomposition following Algorithm 1 in:
    
	Kolda, Tamara Gibson. Multilinear operators for higher-order decompositions.
	United States Department of Eneregy, 2006

	Parameters
    ----------
    data: data tensor of tf or np type
    
    ranks: list or int, desired rank of core tensor. If int
           creates a list repeated order times

    shape: list, rank of original data tensor
    
    stop_thresh: thresh hold for stopping the algorithm
    
    epochs: maximum number of iterations?
    
    dtype: type of numbers in the tensor
    
    init: how the factor matricies will be initiated. If 'unif'
          they are initiated using the uniform dist. If 'norm'
          they are initiated using the normal dist. Default
          is using HOSVD

    limits: if unif, limits for lower and upper value
            if norm, mean and std

    """

	def __init__(self, shape, ranks, stop_thresh = 1e-10, X_data = None,
    	         dtype = tf.float64, init = 'unif', epochs = 1000,
    	         limits = (0,1)):
        
        # need to read up on the difference of constant and variable
		self.order = len(shape)
		self.ranks = ranks if (type(ranks) is list) else [ranks]*self.order
		self.shape = shape
		self.stop_thresh = stop_thresh
		self.init = init
		self.epochs = epochs
		self.dtype = dtype
		self.a = limits[0]
		self.b = limits[1]
		self.init_components(init, X_data)
		self.init_reconstruct()

	def init_components(self, init, X_data): 
		"""
		Init component matricies using HOSVD, should accomodate
		a random init, look into later
		"""
		self.G = tf.Variable(tf.random_uniform(self.ranks, self.a, self.b, self.dtype), 
        	                 name = "G")

		with tf.name_scope('U'):
			self.U = [None] * self.order
			init_val = None

			for n in range(self.order):
				if init == 'hosvd':
					init_val = top_components(X_data, self.ranks[n], n)
        	    
				elif init == 'unif':
					shape = (self.shape[n], self.ranks[n])
					init_val = np.random.uniform(low = self.a, high = self.b, size = shape)
                    
				elif init == 'normal':
					shape = (self.shape[n], self.ranks[n])
					init_val = np.random.normal(loc = self.a, scale = self.b, size = shape)
        		
				self.U[n] = tf.Variable(init_val, name = str(n), dtype = self.dtype)
    
	def init_reconstruct(self):
		"""
		Initialize variable for reconstructed tensor X with components U
		"""
		G_to_X = self.G
		shape = self.ranks[:]

		for n in range(self.order):
			shape[n] = self.shape[n]
			name = None if (n < self.order - 1) else 'X'

			Un_mul_G = tf.matmul(self.U[n], unfold_tf(G_to_X, n))
			with tf.name_scope(name):
				G_to_X = refold_tf(Un_mul_G, shape, n)

		self.X = G_to_X

	def get_core_op(self, X_var):
		"""
		Return tf op used to assign new value to core tensor G.
		G = X times1 u1 times 2 ...
		"""
		X_to_G = tf.identity(X_var)
		shape = list(self.shape)

		for n in range(self.order):
			# the nth dimension of X_to_G will be 
			# the nth desired rank as specified on 
			# instancesation
			shape[n] = self.ranks[n]

			Un_mul_X = tf.matmul(tf.transpose(self.U[n]), unfold_tf(X_to_G, n))
			X_to_G = refold_tf(Un_mul_X, shape, n)

		return tf.assign(self.G, X_to_G)


	def hosvd(self, X_data):
		"""
		HOSVD for a better first guess, before doing HOOI
		"""
		X_var = tf.Variable(X_data, dtype = self.dtype)

		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)
			# why would you shuffle the iterator?
			#for n in shuffled(trange(self.order)):
			for n in trange(self.order):
				# do singular value decomposition on the X_n matrix
				# kolda uses the eigenvalues here, note sure which
				# should be used

				# 'svd%3d' % n has a white space which is not comp of scope name
				string_name = 'svd%3d' % n
				string_name = string_name.replace(" ","")
				_,u,_ = tf.svd(unfold_tf(X_var, n), name = string_name)

				# set U[n] to the first ranks[n] left-singular values of X
				new_U = tf.transpose(u[:self.ranks[n]])
				# to assign to the X_var tf.variable
				svd_op = tf.assign(self.U[n], new_U)

				# run SVD and assign new variable U[n]
				sess.run([svd_op], feed_dict = {X_var : X_data})

				# log fit after training
				X_pred = sess.run(self.X)
				fit = get_fit(X_data, X_pred)
				_log.debug('[U%3d] fit : %.5f' % (n, fit))

			# Compute new core tensor value G
			core_op = self.get_core_op(X_var)
			sess.run([core_op], feed_dict = {X_var : X_data})

			# Log final fit
			X_pred = sess.run(self.X)
			fit = get_fit(X_data, X_pred)
			_log.debug('[G] fit: %.5f' % fit)

			return X_pred


	def get_ortho_iter(self, X_var, n):
		"""
		get SVD for G tensor-product with all U except U[n]
		"""
		Y = tf.identity(X_var)
		shape = list(self.shape)
		idxs = [n_ for n_ in range(self.order) if n_ != n]

		for n_ in idxs:
			shape[n_] = self.ranks[n_]
			name = None if (n_ < idxs[-1]) else 'Y%3d' % n_

			Un_mul_X = tf.matmul(tf.transpose(self.U[n_]), unfold_np(Y, n_))
			with tf.name_scope(name):
				Y = refold_tf(Un_mul_X, shape, n_)
        
		return Y

	def hooi(self, X_data):
		"""
		Higher-Order Orthogonal Iteration
		"""
		X_var = tf.Variable(X_data)
		init_op = tf.global_variables_initializer()
		svd_ops = [None] * self.order

		with tf.Session() as sess:
			sess.run(init_op)
			for e in trange(self.epochs):

				# set up orth iteration ops
				for n in range(self.order):
					Y = self.get_ortho_iter(X_var, n)
					_,u,_ = tf.svd(unfold_tf(Y, n), name = "svd%3d" % n)
					svd_ops[n] = tf.assign(self.U[n], u[:, :self.ranks[n]])
                
				# shuffled or no?
				for n in trange(self.order):
					sess.run([svd_ops[n]], feed_dictv = {X_var : X_data})

					X_pred = sess.run(self.X)
					fit = get_fit(X_data, X_predict)
					_log.debug('[%3d - U%3d] fit: %.5f' % (e,n,fit))

		# Compute new core tensor value G
		core_op = self.get_core_op(X_var)
		sess.run([core_op], feed_dict = {X_var : X_data})

		# Log final fit
		X_predict  = sess.run(self.X)
		fit = get_fit(X_data, X_predict)
		_log.debug('[G] fit: %.5f' % fit)

		return X_predict