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

def covariance_matrix(dim, seed = 1234, diagonal = True, uniform_params = [0,1],
	degree_of_collinearity = 0.5):
	"""
	Function to generate covariance matrix for the 
	multivariate normal distribution

	Params
	------
	dim = nrow & ncol
	seed = random seed
	diagonal = if false, creates multicollinearity
	uniform_params = [lower value, higher value] for uniform
	degree_of_collinearity = avarage number of non-zero off diagonals 
							 if diagonal is set to false
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
								size = (dim**2)).reshape(5,5)
		
		return np.matmul(cov, np.transpose(cov))
		
		 
		

