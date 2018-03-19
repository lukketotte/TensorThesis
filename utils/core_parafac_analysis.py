import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from utils_np import *
from math import ceil

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from decompositions.parafac_np import parafac

"""
Methods for data analysis for this master thesis
which mainly concerns the relationship of PARAFAC
on the original data and core tensor
"""

def error_parafac(tensor, max_rank, init = "hosvd", verbose = False):
	"""
	Returns a vector of len max_rank which contains the 
	converged error rate for all the ranks up till max_rank
	from training PARAFAC on the tensor 

	Params
	------
	tensor: np.ndarray, the data
	max_rank: int, run parafac from cp rank 1 to max_rank
	init: str, initiaztion method for factor matricies
	"""
	error_over_ranks = [None] * max_rank
	# initiate class
	pc = parafac(init = init)
	pc.X_data = tensor

	for rank in range(max_rank):
		if verbose:
			print("Current rank: %d" % (rank + 1))
		pc.rank = rank + 1
		pc.init_factors()
		error_temp = pc.parafac_als()
		# store last term in error_over_ranks
		error_over_ranks[rank] = error_temp[len(error_temp) - 1]

	return error_over_ranks


def to_image(tensor):
	"""
	Convience function for analysis of images 
	where float has to converted back to uint8 
	for valid plots

	Params
	------
	tensor: np.ndarray or tl.tensor
	"""
	# if np.ndarray no transform
	if isinstance(tensor, np.ndarray):
		im = tensor
	# if tl.tensor transform to np.ndarray
	else:
		im = tl.to_numpy(tensor)
	im -= im.min()
	im /= im.max()
	im *= 255
	return im.astype(np.uint8)

def covariance_matrix(dim, seed = 1234, diagonal = True, uniform_params = [0,1]):
	"""
	Function to generate covariance matrix for the 
	multivariate normal distribution. Here we consider
	multicollinearity within the factor matricies

	Params
	------
	dim = nrow & ncol
	seed = random seed
	diagonal = if false, creates multicollinearity
	uniform_params = [lower value, higher value] for uniform
	"""
	if diagonal:
		cov = np.diag(np.random.uniform(low = uniform_params[0], 
									high = uniform_params[1],
									size = dim))
		return cov
	else:
		# to get a symmetric positive definite matrix,
		# it will be generated as cov*cov'
		cov = np.random.uniform(low = uniform_params[0], 
								high = uniform_params[1],
								size = (dim**2)).reshape(dim,dim)
		
		return np.matmul(cov, np.transpose(cov))
		
		 
def covariance_matrix_parafac(dim, pc_rank, dependency_structure, 
	seed = 1234, uniform_params = [0,1]):
	"""
	This function returns all the factor matricies, which will be generated
	assuming dependencies between their columns. That is in terms of the outer
	product

	Params
	------
	dim = [int], elements in the modes of tensor
	pc_rank = int, pc rank of tensor
	dependency_structure = [int]*3, 1's or 0's depending on whether a element in the 
						   covariance matrix of MVND should be zero or not.
						   [12, 13, 23]
	seed = int, random seed
	uniform_params = [int], [lower value, higher value] for uniform
	"""
	# only third order tensors in simulation. If an int is passed make it into a list
	if isinstance(dim, int):
		dim = [dim]*3

	N = len(dim)
	factor_matricies = [None]*N

	for n in range(N):
		factor_matricies[n] = np.zeros((dim[n], pc_rank))

	cov = covariance_matrix(dim = 3, diagonal = False)
	# position [0,1], [1,0]
	cov[0,1] *= dependency_structure[0]
	cov[1,0] = cov[0,1]
	# position[0,2], [2,0]
	cov[0,2] *= dependency_structure[1]
	cov[2,0] = cov[0,2]
	# position [1,2], [2,1]
	cov[1,2] *= dependency_structure[2]
	cov[2,1] = cov[1,2]

	for i in range(pc_rank):
		dim_max = max(dim)
		columns_of_matricies = np.random.multivariate_normal([0]*3, cov, size = dim_max)
		# assign to the factor matricies
		for n in range(N):
			factor_matricies[n][:,i] = columns_of_matricies[0:dim[n], n]

	
	return factor_matricies



	
