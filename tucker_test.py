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

	Slight modification is that the singular values will be used, but
	that is of no matter for symmetric matricies

	Parameters
	--------------
	shape: [int], shape of input data (X_data)

	ranks: [int] or int, desired shape of core tensor

	X_data: np.ndarray, data of numpy type

	dtype: type of data in tensors

	init: str, type of initization for the component matricies
	      possible values: 'hosvd', 'random'

	epochs: int, number of iterations

	limits: (int), if random init will denote the upper and lower limit (uniform)

	"""

	def __init__(self, shape, ranks, X_data, dtype = tf.float64,
		         init = 'hosvd', epochs = 100, limits = (0,1),
		         stop_thresh = 1e-10):

		self.X = tf.get_variable("X", dtype = dtype, initializer = X_data)
		# or rank, in tensorflow language
		self.order = len(shape)
		self.ranks = ranks if (type(ranks) is list) else [ranks]*self.order
		self.dtype = dtype
		self.init = init
		self.epochs = 100
		self.a = limits[0]
		self.b = limits[1]
		self.stop_thresh = stop_thresh
		# initialize the core tensor and component matricies
		self.init_components(init, X_data)

			
	def init_components(self, init, X_data):
		"""
		init components using either HOSVD or uniform (a, b]
		"""
		# initialize core tensor as tf.Variable with zeroes. Won't
		# be touched until the final step
		self.G = tf.get_variable("G", self.ranks, dtype = self.dtype,
			initializer = tf.zeros_initializer)
		# initialize An as the rank[n] leading singular values
		with tf.name_scope('A'):
			self.A = [None] * self.order
			init_val = None

			for n in range(self.order):
				if init == 'hosvd':
					# calculate the singular values of Xn * Xn'
					Xn = unfold_tf(self.X, n)
					Y = tf.matmul(Xn, tf.transpose(Xn))
					# want An as 0, ... , ranks[n] first singular values
					# note that tf.svd returns the singular values in 
					# descending order. 
					init_val = tf.svd(Y, compute_uv = False)[:self.ranks[n]]
					init_val = tf.diag(init_val)
				elif init == 'unif':
					shape = (self.shape[n], self.ranks[n])
					init_val = np.random.uniform(low = self.a, 
						                         high = self.b, size = shape)
				name_str = "A%d" % n
				name_str = name_str.replace(" ","")
				
				self.A[n] = tf.get_variable(name_str, dtype = self.dtype, 
					                        initializer = init_val)
				print(self.A[n])
				# <tf.Variable 'A2:0' shape=(ranks[n],) dtype=float64_ref>

	# TODO: method for updating core tensor, possible to use 
	# the methods in utils I guess

	def tucker_update(X_var, A):
		"""
		This method updates the core tensor defined in the 
		init_components method as G.

		Here the kruskal method from utils is used
		"""
		return tf.assign(self.G, kruskal(X_var, self.A))