import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *
import logging

logging.basicConfig(filename = 'loss_parafac.log', level = logging.DEBUG)
_log = logging.getLogger('decomp')

class parafac():
	"""
	Algorithm 2 in:
    
	Kolda, Tamara Gibson. Multilinear operators for higher-order decompositions.
	United States Department of Eneregy, 2006

	Paramaters:
	----------
	X_data: numpy.ndarray, data tensor

	shape: [int] shape of X_data
	
	rank: int or None, number of factors if int, else the rank is determined 
	      within the algorithm. Start by assuming it to be int, None feature
	      might be added at later stage

	epochs: int, number of iterations
	
	stop_thresh: double, min change before stopping
	
	init: str, initiation for factor matricies, random or hosvd. Sticking with
		  random for now, something is off with the pseudo code in Kolda
	"""
	def __init__(self, X_data = None, shape = None, rank = None, epochs = 8,
		         stop_tresh = 1e-10, dtype = tf.float64, init = 'random', limits = [0,1]):
		self.epochs = epochs
		self.stop_tresh = stop_tresh
		self.dtype = dtype
		self.init = init
		
		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._rank = rank
		self.U = None
		self._limits = limits


	@property
	def X_data(self):
		# print("getting X")
		return self._X_data
	@X_data.setter
	def X_data(self, X):
		# print("setting X")
		if isinstance(X, np.ndarray):
			# set the shape and order of tensor 
			self._shape = list(X.shape)
			self._order = len(self._shape)
			self._X_data = tf.get_variable("X_data", dtype = self.dtype,
				initializer = X)
		else:
			raise TypeError("Has to be of numpy.ndarray type")

	@property
	def rank(self):
		return self._rank
	@rank.setter
	def rank(self, r):
		if isinstance(r, int):
			if r >= 0:
				self._rank = r
			else:
				raise ValueError("Rank has to be positive")
		else:
			raise TypeError("Rank has to be int")

	@property
	def limits(self):
		return self._limits
	@limits.setter
	def limits(self, lst):
		if isinstance(lst, list):
			if isinstance(lst[0], int):
				self._limits = lst
			raise TypeError("Expecting [int]")
		else:
			raise TypeError("Expecting list")

	def init_factors(self):
		"""
		
		"""
		if not isinstance(self._rank, type(None)):
			if not isinstance(self._X_data, type(None)):
				# list of factor matricies
				self.U = [None] * self._order
				self.B = [None] * self._order
				init_val = None

				if(self.init == "hosvd"):
					for n in range(self._order):
						Xn = unfold_tf(self._X_data, n)
						_, init_val,_ = tf.svd(Xn, compute_uv = True)
						init_val = tf.slice(init_val, begin = [0,0], size = [self._shape[n], self._rank])
						print(init_val.get_shape())
						print(type(init_val))
						# normalize over the columns of init_val
						# init_val = tf.nn.l2_normalize(init_val, 1)
						# self.U[n] = tf.get_variable(name = str(n), dtype = self.dtype, initializer = init_val)
						self.U[n] = init_val


				elif(self.init == "random"):
					for n in range(self._order):
						self.U[n] = tf.get_variable(name = str(n), dtype = self.dtype, 
							initializer = tf.random_uniform(minval = self._limits[0], maxval = self._limits[1],
															shape = [self._shape[n], self._rank], dtype = self.dtype))
						# normalize the columns 
						self.U[n] = tf.nn.l2_normalize(self.U[n], 0)
						# B[n] = U[n]'U[n] which is R by R
						self.B[n] = tf.matmul(tf.transpose(self.U[n]),self.U[n])
						# print(self.B[n].get_shape())

				else:
					raise ValueError("%s not valid for init paramater (hosvd or random)" % self.init)
			else:
				raise TypeError("X_data needs to be set prior to initiaztion of "+
					"factor matricies")
		else:
			raise TypeError("Desired rank (number of factors) has to be set prior "+
				"to initiaztion")

	def get_factor_matricies(self):
		return self.U

	def get_B(self):
		return self.B

	def reconstruct_X_data(self):
		# using the mode-0 unfolding:
		# Xn = An(khatri rao of all the other factor matricies)
		A_khatri = kruskal_tf_parafac(self.U[:0] + self.U[1 :])
		X0 = tf.matmul(self.U[0], tf.transpose(A_khatri))
		return refold_tf(X0, self._shape, 0)

	def parafac_ALS(self):
		if not isinstance(self.U, type(None)):
			if not isinstance(self._X_data, type(None)):
				init_op = tf.global_variables_initializer()
				with tf.Session() as sess:
					sess.run(init_op)

					# ------- ALS algorithm ------- #
					# TODO: include stopping criterion based on error
					for e in trange(self.epochs):
						for n in range(self._order):
							# Calculate V = hadamard product of all B but Bn
							# will look a little different depending on whether
							# n is 0 or not
							if n == 0:
								V = self.B[1]
								# skip first two elements in B
								for B in self.B[2: ]:
									V = tf.multiply(V, B)
							elif not n == 0:
								V = self.B[0]
								for B in self.B[1 :] :
									if not B == n:
										V = tf.multiply(V, B)

							xn = unfold_tf(self._X_data, n)
							# Calculate the khatri rao prod of all 
							# factor matricies but the nth entri
							khatri_u = kruskal_tf_parafac(self.U[:n] + self.U[n+1 :])
							# print(xn.get_shape())
							# print(khatri_u.get_shape())
							# print(mpinv(V).get_shape())


							# update nth factor matrix
							self.U[n] = tf.matmul(xn, tf.matmul(khatri_u, mpinv(V)))

							# if n neq (self._order - 1) normalize columns of factor matrix
							if not n == (self._order - 1):
								self.U[n] = tf.nn.l2_normalize(self.U[n], 0)

							self.B[n] = tf.matmul(tf.transpose(self.U[n]), self.U[n])

			else:
				raise TypeError("Need to set X_data prior to ALS")
		else:
			raise TypeError("Need to run init_factors() prior to ALS")
