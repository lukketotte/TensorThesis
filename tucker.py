import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *

class TuckerDecomposition():
    """
    Used for Tucker decomposition following Algorithm 1 in:
    
    Kolda, Tamara Gibson. Multilinear operators for higher-order decompositions.
    United States Department of Eneregy, 2006

    Parameters
    ----------
    data: data tensor of tf or np type
    
    ranks: list or int, desired rank of core tensor. If int
           creates a list repeated order times

    shape: list, rank of original data tensor
    
    stop_thresh: thresh hold for stopping the algorithm
    
    epochs: maximum number of iterations?
    
    dtype: type of numbers in the tensor
    
    init: how the factor matricies will be initiated. If 'unif'
          they are initiated using the uniform dist. If 'norm'
          they are initiated using the normal dist. Default
          is using HOSVD

    limits: if unif, limits for lower and upper value
            if norm, mean and std

    """

    def __init__(self, X_data, shape, ranks, stop_thresh = 1e-10, 
    	         dtype = "float64", init = 'hosvd', epochs = 1000,
    	         limits = (0,1)):
        
        # need to read up on the difference of constant and variable
        self.X = tf.constant(X_data, dtype = dtype)
        self.order = len(shape)
        self.ranks = ranks if (type(ranks) is list) else [ranks]*self.order
        self.stop_thresh = stop_thresh
        self.init = init
        self.epochs = epochs
        self.init_components(init, X_data)
        self.a = limits[0]
        self.b = limits[1]

    def init_components(self, init, X_data): 
        """
        Init component matricies using HOSVD, should accomodate
        a random init, look into later
        """
        with tf.name_scope('U'):
        	self.U = [None] * self.order

        	for n in range(self.order):
        		if init == 'hosvd':
        			init_val = top_components(X_data, self.ranks[n], n)
        		elif init == 'unif':
        			shape = (self.shape[n], self.ranks[n])
        			init_val = np.random.uniform(low = self.a, high = self.b, size = shape)
        		self.U[n] = init_val
        		elif init == 'normal'




