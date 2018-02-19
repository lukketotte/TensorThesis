import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *
import logging

logging.basicConfig(filename = 'loss.log', level = logging.DEBUG)
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
	
	init: str, initiation for factor matricies, random or hosvd
	"""
	def __init__(self, X_data = None, shape = None, rank = None, epochs = 50,
		         stop_tresh = 1e-10, dtype = tf.float64, init = 'hosvd'):
		self.epochs = epochs
		self.stop_tresh = stop_tresh
		self.dtype = dtype
		self.init = init

		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._rank = rank
		self.U = None


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

	def init_factors(self):
		"""
		
		"""
		if not isinstance(self._rank, type(None)):
			if not isinstance(self._X_data, type(None)):
				# list of factor matricies
				self.U = [None] * self._order
				self.B = [None] * self._order
				init_val = None
				self.zero_fill_row = [None] * self._order
				self.zero_fill_col = [None] * self._order

				if(self.init == "hosvd"):
					for n in range(self._order):
						Xn = unfold_tf(self._X_data, n)
						Xn = tf.matmul(Xn, tf.transpose(Xn))
						# only keep the rank number of singular values
						init_val = tf.svd(Xn, compute_uv = False)[:self._rank]
						init_val = tf.diag(init_val)
						# the number of rows of init_val need to be the same
						# as the nth value of self._shape
						if not init_val.get_shape()[0] == self._shape[n]:
							# have to append zeroes to make matrix multiplication
							# in ALS defined. Does the fill matrix have to be 
							# a tf.variable?
							fill_name = "0%d" % n
							# get a white space in str, not allowed in
							# tf.variable name. 
							fill_name = fill_name.replace(" ", "")
							# no need to do this all over again in the ALS algorithm, so 
							# store it in list. In ALS simply check if position is None or not
							# TODO: check if this really shouldnt be constant, can you use 
							#		a variable as initializer for another variable
							self.zero_fill_row[n] = tf.get_variable(fill_name, ((self._shape[n] - self._rank), self._rank),
								dtype = self.dtype, initializer = tf.zeros_initializer)
							init_val = tf.concat([init_val, self.zero_fill_row[n]], 0)

						self.U[n] = tf.get_variable(name = str(n), dtype = self.dtype, initializer = init_val)


				elif(self.init == "random"):
					for n in range(self._order):
						self.U[n] = tf.get_variable(tf.random_uniform(self.ranks, 0, 1, self.dtype), 
							name = str(n))

				# No matter initialization type, normalise columns of U[n] and
				# assign U[n]' * U[n] to B[n]

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