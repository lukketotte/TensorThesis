import numpy as np
from utils_np import *

class tucker():
	"""
	Computes the tucker decomposition of a tensor using ALS

	Parameters:
	-----------
	X_data: raw data tensor
	shape: shape of input data
	rank: shape of core tensor
	"""

	def __init__(self, X_data = None, shape = None, ranks = None, epochs = 1000, 
				 stop_thresh = 1e-12, init = 'hosvd'):

		self.epochs = epochs
		self.stop_thresh = stop_thresh
		self.init = init

		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._ranks = ranks
		self.A = None	
	
	@property
	def X_data(self):
		print("getting X")
		return self._X_data
	@X_data.setter
	def X_data(self, X):
		print("setting X")
		if isinstance(X, np.ndarray):
			# set the shape and order of tensor 
			self._shape = list(X.shape)
			self._order = len(self._shape)
			self._X_data = X
		else:
			raise TypeError("Has to be of numpy.ndarray type")

	@property
	def ranks(self):
		return self._ranks
	@ranks.setter
	def ranks(self, lst_rank):
		if isinstance(lst_rank, list) and isinstance(lst_rank[0], int):		
			self._ranks = lst_rank		
		else: 
			raise ValueError("Has to be list of integers")

	def get_component_mats(self):
		return self.A

	def init_components(self):
		"""
		core: ndarray, core tensor of Tucker decomposition
		factors: [ndarray], list of factors for decomp with
				 core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes

		shape of core tensor assumed to be same as data tensor if not specified
		prior to running
		"""
		# if no ranks has been given, assume dimension of core is that of data tensor
		if isinstance(self._ranks, type(None)):
			self._ranks = [self._shape[mode] for mode in self._order]
			print("WARNING: Core tensor ranks not set, assumed to be same as X_data")

		# check for input validation
		if not isinstance(self._X_data, type(None)):
			# [component mats]
			self.A = [None] * self._order

			if self.init is 'ranom':
				# just keep it as 0,1 init
				# nth component matrix is In by Jn (Kolda & Bader)
				for mode in range(self._order):
					init_val = np.random.uniform(low = 0, high = 1,
						size = self._shape[mode] * self._ranks[mode])
					init_val = init_val.reshape(self._shape[mode], self._ranks[mode])
					self.A[mode] = init_val

			elif self.init is 'hosvd':
				for mode in range(self._order):
					init_val, rank_higher = hosvd_init(self._X_data, mode, self._ranks[mode])
					# this is a hack really, but not sure what else
					# to do if tucker rank exceeds the columns of n-mode
					# unfolding
					if rank_higher:
						diff = self._ranks[mode] - self._shape[mode]
						fill = np.random.uniform(low = 0, high = 1,
							size = diff).reshape(self._shape[mode], diff)
						init_val = np.concatenate((init_val, fill), 1)
					self.A[mode] = init_val

		else:
			raise TypeError("X_data needs to be set prior to init_components()")

	def partial_tucker(self):
		if not isinstance(self.A, type(None)):
			if not isinstance(self._X_data, type(None)):
		
				rec_errors = []
				norm_x = norm(self._X_data, 2)

				for e in range(self.epochs):
					for index in range(self._order):
						core_approximation = multi_mode_dot(self._X_data, self.A, 
															skip=index, transpose=True)

						eigenvecs, _, _ = partial_svd(unfold_np(core_approximation, index), n_eigenvecs=self._ranks[index])
						self.A[index] = eigenvecs
					
					core = multi_mode_dot(self._X_data, self.A, transpose=True)
					
					# The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
					rec_error = np.sqrt(abs(norm_x**2 - norm(core, 2)**2)) / norm_x
					rec_errors.append(rec_error)

					if e > 1:
						if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
							print('converged in {} iterations.'.format(e))
					break

				return core

			else:
				raise TypeError("Set X_data prior to running ALS") 
		else:
			raise TypeError("Run init_components() prior to ALS")

