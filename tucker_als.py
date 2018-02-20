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

	def __init__(self, X_data = None, shape = None, rank = None, epochs = 50, 
				 stop_thresh = 1e-22, dtype = tf.float64):

		self.epochs = epochs
		self.stop_thresh = stop_thresh
		self.dtype = dtype

		self._X_data = X_data
		self._shape = shape
		self._order = len(shape) if (type(shape) is list) else None
		self._rank = rank
		# will become list of component matricies in init_component() method
		# if not set will throw an exception in the ALS algorithm
		self.U = None
		# if rank is set on instance, need to initialize G
		if not isinstance(rank, type(None)):
			self.G = tf.get_variable("G", rank, dtype = dtype, 
				initializer = tf.zeros_initializer)

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
			self._X_data = tf.get_variable("X_data", dtype = self.dtype,
				initializer = X)
			# when X_data is set, the component matricies U, V, W (in 3d case)
			# will be initialized
			self.init_components()
		else:
			raise TypeError("Has to be of numpy.ndarray type")

	@property
	def rank(self):
		return self._rank
	@rank.setter
	def rank(self, lst_rank):
		if isinstance(lst_rank, list) and isinstance(lst_rank[0], int):
			
			self._rank = lst_rank
			# set core tensor:
			self.G = tf.get_variable("G", lst_rank, dtype = self.dtype, 
				initializer = tf.zeros_initializer)
		
		else: 
			raise ValueError("Has to be list of integers")

	def init_components(self):
		"""
		In the 3d case:
		U = r1 principal right singular vectors of X1
		V = r2 principal right singular vectors of X2
		W = r3 principal right singular vectors of X3
		"""
		if not isinstance(self._order, type(None)):
			# print("Initializing the component matricies")
			# list that will hold the component matricies
			# associated with each unfolding of the data tensor
			self.U = [None] * self._order
			
			if not isinstance(self._rank, type(None)):

				for n in range(self._order):
					# U[n] <- leading rank[n] right principal vectors
					# of Xn
					Xn = unfold_tf(self._X_data, n)
					# setting comupte_uv = True returns singular values, left sing vecs,
					# right sing vecs in that order
					init_val = tf.svd(tf.matmul(Xn, tf.transpose(Xn)), compute_uv = True)[2]
					# take rank[n] columns singular vectors 
					init_val = init_val[: , :self._rank[n]]
					self.U[n] = tf.get_variable(name = str(n), dtype = self.dtype, initializer = init_val)

			else:
				raise TypeError("Expecting rank to be [int] but gets None, set rank")
		else:
			raise TypeError("Expecting order to be int but gets None, set X_data")

	def return_component_matricies(self):
		"""
		Method for accessing U 
		"""
		return self.U

	def G_to_X(self):
		"""
		Reconstructs an estimate of the data tensor using G, and the 
		component matricies

		Gn * An -> refold as [I1,...,In-1, rn, ... , rN]
		"""
		if not isinstance(self.G, type(None)):
			temp_G = self.G
			for n in range(self._order):
				# the refolded tensor shape after the nth step
				# of the kruskal operator as defined by kolda
				refold_shape = self._shape[:(n+1)] + self._rank[(n+1):]
				# print(refold_shape)
				temp_G = unfold_tf(temp_G, n)
				print(temp_G.get_shape())
				temp_G = tf.matmul(self.U[n], temp_G)
				temp_G = refold_tf(temp_G, refold_shape, n)
			
			return(temp_G)




		else:
			raise TypeError("tucker_ALS() has to be run prior to estimation")


	def tucker_ALS(self):
		"""
		Runs the iterative ALS scheme, continously updating the component matricies
		until either (a) iterations = epocs or (b) negligeble change in frobenius norm
		of (V kron W)' X1'U
		"""
		if not isinstance(self._X_data, type(None)):
			# focus on 3d case for now
			if self._order == 3:
				# init components must have been set before 
				# using the algorithm
				if not isinstance(self.U, type(None)):
					init_op = tf.global_variables_initializer()
					svd_op = [None] * self._order

					with tf.Session() as sess:
						sess.run(init_op)

						# ------------- ALS ALGORITHM -------------#

						# Update step, define the operation:
						for e in trange(self.epochs):

							# set up the stopping criterion
							G1_norm = tf.matmul(update_component_matricies(self.U, self._X_data, 0), self.U[0])
							for n in range(self._order):
								### set up TF graph ###
								temp = update_component_matricies(self.U, self._X_data, n)
								# When n = 0 returns:
								# (U[1] kronecker U[2])' * X0'
								
								# Take the rank[n] columns of right sing vecs
								temp_v = tf.svd(temp, compute_uv = True)[2]

								# NO UPDATE MADE WITH FORMULATION AS BELOW
								# svd_op[n] = tf.assign(self.U[n], temp_v[:, :self._rank[n]])
								# sess.run(svd_op[n])
								self.U[n] =  temp_v[:, :self._rank[n]]
								
							# check change in norm
							G1_temp = tf.matmul(update_component_matricies(self.U, self._X_data, 0), self.U[0])
							delta_norm = ((G1_norm.eval() - G1_temp.eval())**2).sum()
							# log the change in norm
							_log.debug('Change in norm iter %d: %.10f' %(e, delta_norm))
							# break if stop_tresh is met
							#if delta_norm <= self.stop_thresh:
							#	print("\n Change in norm = %.5f, stop threshhold met" % delta_norm)
							#	break

						# -----------------------------------------#
						# return object, might be off in the scope but should be correct
						G1 = tf.matmul(update_component_matricies(self.U, self._X_data, 0), self.U[0])
						self.G = refold_tf(G1, self._rank, 0)
						return(G1)
				else:
					raise TypeError("need to run init_components() beofre ALS")
			else:
				 raise ValueError("Currently only valid for 3d tensor")
		else:
			raise TypeError("Data tensor has not been set")
