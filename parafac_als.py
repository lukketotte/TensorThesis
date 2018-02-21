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
	
	row_info: str, added into the debug file, specifying whether parafac has
			  been run on original data or a core tensor
	"""
	def __init__(self, X_data = None, shape = None, rank = None, epochs = 10,
		         stop_thresh = 1e-5, dtype = tf.float64, init = 'random', limits = [0,1],
		         row_info = None):
		self.epochs = epochs
		self.stop_thresh = stop_thresh
		self.dtype = dtype
		self.init = init
		
		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._rank = rank
		self.U = None
		self._limits = limits
		self._row_info = row_info


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

	@property
	def row_info(self):
		return self._row_info
	@row_info.setter
	def row_info(self, string):
		if isinstance(string, str):
			if string is "original" or string is "core":
				self._row_info = string
			else:
				raise ValueError("Only 'original' and 'core' are valid entries for row_info")
		else:
			raise TypeError("Must be string")

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
						# not that the left singular matrix is In by In
						if self._rank <= self._shape[n]:
							init_val = tf.slice(init_val, begin = [0,0], size = [self._shape[n], self._rank])
						# in the case where rank > shape[n]:
						elif self._rank > self._shape[n]:
							init_val = tf.slice(init_val, begin = [0,0], size = [self._shape[n], self._shape[n]])
							# append random uniform to init_val with In rows and rank - In columns
							uni_string = "u%d" % n
							uni_string = uni_string.replace(" ","")
							uni_fill = tf.get_variable(name = uni_string, dtype = self.dtype,
								initializer = tf.random_uniform(minval = self._limits[0], maxval = self._limits[1],
																shape = [self._shape[n], self._rank - self._shape[n]],
																dtype = self.dtype))
							init_val = tf.concat([init_val, uni_fill], 1)

						self.U[n] = init_val
						self.B[n] = tf.matmul(tf.transpose(self.U[n]),self.U[n])


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
					for e in trange(self.epochs):						
						for n in range(self._order):
							# create the n unfolding of X
							xn = unfold_tf(self._X_data, n)
							# create list that exludes U[n]
							temp_lst = self.U[:n] + self.U[(n+1):]
							# reversed order
							temp_kahtri = kruskal_tf_parafac(temp_lst[::-1])
							self.U[n] = tf.matmul(xn, tf.transpose(mpinv(temp_kahtri)))

						# check fit, log it. Do it at start of iteration rather than the end
						fit = get_fit(unfold_tf(self._X_data,0).eval(), unfold_tf(self.reconstruct_X_data(),0).eval())
						_log.debug('PARAFAC, %d, %d, %.10f, %s' %(self._rank,e, fit, self._row_info))
						if not e == 0:
							if abs(fit) <= self.stop_thresh:
								print("\nfit: %.5f. Breaking." %fit)
								break

			else:
				raise TypeError("Need to set X_data prior to ALS")
		else:
			raise TypeError("Need to run init_factors() prior to ALS")
