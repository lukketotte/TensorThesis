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
    
    ranks: desired rank of core tensor
    
    stop_thresh: thresh hold for stopping the algorithm
    
    epochs: maximum number of iterations?
    
    dtype: type of numbers in the tensor
    
    init: initialize the factor matricies

    """

    def __init__(self, X_data, ranks, stop_thresh = 1e-10, 
    	         dtype = "float64", init = 'hosvd', epochs = 1000):

        self.X = tf.Variable(X_data, dtype = dtype)
        self.ranks = ranks
        self.stop_thresh = stop_thresh
        self.init = init
        self.epochs = epochs
        self.init_components(init, X_data)
        self.component_matricies = []

    def init_components(self, init, X_data): 



