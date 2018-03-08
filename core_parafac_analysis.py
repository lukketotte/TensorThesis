import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from utils_np import *
from parafac_np import parafac
from math import ceil

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