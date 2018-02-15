import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *
import logging

logging.basicConfig(filename = 'loss.log', level = logging.DEBUG)
_log = logging.getLogger('decomp')

class TuckerDecomposition():
	"""
	Computes the tucker decomposition of a tensor using ALS

	Parameters:
	-----------
	X_data: raw data tensor
	shape: shape of input data
	rank: shape of core tensor
	"""

	def __init__(self, X_data = None, shape = None, rank = None, epochs = 1000, 
				 stop_thresh = 1e-10, dtype = tf.float64):

		self.epochs = epochs
		self.stop_thresh = stop_thresh
		self.dtype = dtype
		
		self.X_data = X_data
		self.shape = shape
		self.order = len(shape) if (type(shape) is list) else None
		self.rank = rank

		@property
		def X_data(self):
			return self.X_data
		@X_data.setter
		def X_data(self, X):
			if isinstance(X, np.ndarray):
				# set the shape and order of tensor 
				self.shape = X.shape
				self.order = len(self.shape)
				self.X_data = tf.get_variable("X_data", dtype = self.dtype,
					initializer = X)
				# when X_data is set, the component matricies U, V, W (in 3d case)
				# will be initializes
				self.init_components()
			else:
				raise ValueError("Has to be of numpy.ndarray type")

		@property
		def rank(self):
			return self.rank
		@rank.setter
		def rank(self, lst_rank):
			if isinstance(lst_rank, list) and isinstance(lst_rank[0], int):
				self.rank = lst_rank
			else: 
				raise ValueError("Has to be list of integers")

		def init_components(self):
