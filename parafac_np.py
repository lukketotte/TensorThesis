import numpy as np
from tqdm import trange
from utils_np import *
import logging
from numpy.linalg import pinv
from numpy.linalg import svd
from sklearn.preprocessing import normalize

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
	def __init__(self, X_data = None, shape = None, rank = None, epochs = 1000,
				stop_thresh = 1e-10, dtype = np.float64, init = 'random', limits = [0,1],
		    	row_info = None):

		self.epochs = epochs
		self.stop_thresh = stop_thresh
		self.dtype = dtype
		self.init = init

		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._rank = rank
		self.A = None
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
			self._X_data = X
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
		random or hosvd
		"""
		if not isinstance(self._rank, type(None)):
			if not isinstance(self._X_data, type(None)):

				self.A = [None] * self._order

				if self.init is "random":
					for mode in range(self._order):
						init_val = np.random.uniform(low = self.limits[0], high = self.limits[1],
										             size = self._shape[mode] * self._rank)

						self.A[mode] = init_val.reshape(self._shape[mode], self._rank)
				
				if self.init is 'hosvd':
					for mode in range(self._order):
						init_val, rank_higher = hosvd_init(self._X_data, mode, self._rank)
						# if rank is larger than mode fill remaining columns with 
						# uniform ()
						if rank_higher:
							diff = self._rank - self._shape[mode]
							fill = np.random.uniform(low = self.limits[0], high = self.limits[1],
													 size = diff * self._shape[mode]).reshape(self._shape[mode], diff)
							init_val = np.concatenate((init_val, fill), 1)

						self.A[mode] = init_val

			else:
				raise TypeError("X_data need to be set")
		else:
			raise TypeError("Rank needs to be set")

	def get_factors(self):
		if not isinstance(self.A, type(None)):
			return self.A
		else:
			raise TypeError("Run init_factors() prior")

	def reconstruct_X(self):
		"""
		Reconstructs the tensor from the mode0 unfolding
		estimate refolded
		"""
		if not isinstance(self.A, type(None)):
			
			khatri_prod = khatri_rao_list(self.A[1 :])
			X0 = np.dot(self.A[0], np.transpose(khatri_prod))
			
			return refold_np(X0, 0, self._shape)
		else:
			raise TypeError("Run init_factors() prior")
	
	def parafac_als(self, return_error = True):
		if not isinstance(self.A, type(None)):
			if not isinstance(self.X_data, type(None)):

				rec_errors = []
				norm_X = norm(self._X_data, 2)

				for e in range(self.epochs):
					for mode in range(self._order):
						pseudo_inverse = np.ones([self._rank, self._rank])

						for i, A in enumerate(self.A):
							if i != mode:
								pseudo_inverse[:] = pseudo_inverse * np.dot(np.transpose(A), A)
						A = np.dot(unfold_np(self._X_data, mode), khatri_rao_list(self.A[:mode] + self.A[(mode+1):]))
						# TODO: catch error of non-singular matrix
						A = np.transpose(np.linalg.solve(np.transpose(pseudo_inverse), np.transpose(A)))
						self.A[mode] = A

					rec_error = norm(self._X_data - self.reconstruct_X(), 2) / norm_X
					rec_errors.append(rec_error)

					if e > 1:
						if abs(rec_errors[-2] - rec_errors[-1]) < self.stop_thresh:
							print("Converged in {} iterations. Error = {}".format(e, rec_error))
							break

				if return_error:
					return rec_errors
			else:
				raise TypeError("X_data has not been set")
		else:
			raise TypeError("Run init_factors() prior")

