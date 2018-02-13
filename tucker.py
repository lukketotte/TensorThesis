import tensorflow as tf
import numpy as np
from tqdm import trange
from utils import *

class TuckerDecomposition():
    """
    Used for Tucker decomposition following Algorithm 1 in:
    
    Kolda, Tamara Gibson. Multilinear operators for higher-order decompositions.
    United States Department of Eneregy, 2006
    """

    def __init__(self, shape, ranks, regularize = 1e-5, 
    	         dtype = tf.float64, init = 'random'):
        