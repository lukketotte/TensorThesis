import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from scipy.linalg import kron

"""
Helper methods for the main classes used in the other
classes.
https://github.com/ebigelow/tf-decompose/blob/master/utils.py
"""

def top_components(X, rank, n):
    """
    Based on algorithm 1 in 
    T. Kolda, 2006, Multilinear Operators for Higher-Order Decompositions, Sandia
    Formula is A^(n) <- J_n leading eigenvalues of X_n * X_n'

    Important note: any tf.tensor returned by .eval() is a numpy.ndarray
    """
    # for a 2d tensor the unfolding leaves the matrix intact so no
    # need to do any input validation in terms of shape
    X_ = unfold_np(X, n)
    A = X_.dot(X_.T)
    # A is square symmetric
    N = A.shape[0]
    # take the N-rank to N-1 eigenvalues
    _, U = eigh(A, eigvals = (N - rank, N - 1))
    # ::-1 returns the reversed order
    U = np.array(U[: , ::-1])
    return U

def unfold_np(arr, ax):
    """
    unfold np array along its ax-axis
    from: https://gist.github.com/nirum/79d8e14da106c77c02c1
    """
    return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def unfold_tf(X, n):
    """
    unfold TF variable X along the n-axis
    input shape: (I_1, ..., I_N)
    output shape: (d_n, I_1 * ... * I_n-1 * I_n+1 * ... * I_N)
    """
    shape = X.get_shape().as_list()
    # subtract one from each element
    idxs = [i for i,_ in enumerate(shape)]

    new_idxs = [n] + idxs[:n] + idxs[(n+1):]
    B = tf.transpose(X, new_idxs)

    dim = shape[n]

    return tf.reshape(B, [dim, -1])

def refold_tf(X, shape, n):
    """
    X: tf variable
    shape: desired shape of output tensor
           for 3d [matricies, rows, cols]
    n: assume X is unrolled along shape[n]
    """
    # subtracts one from each element in shape list
    idxs = [i for i,_ in enumerate(shape)]

    shape_temp = [shape[n]] + shape[:n] + shape[(n+1):]
    B = tf.reshape(X, shape_temp)
    new_idxs = idxs[1:(n+1)] + [0] + idxs[(n+1):]
    return tf.transpose(B, new_idxs)

def get_fit(X, Y):
    """
    Compute squared frobenius norm:
      ||X - Y||_F^2  = <X,X> + <Y,Y> - 2 <X,Y>
      run within a tf.Session() so we have numpy arrays
    """
    normX = (X ** 2).sum()
    normY = (Y ** 2).sum()
    norm_inner = (X * Y).sum()

    norm_residual = normX + normY - norm_inner
    # fit as percentage, lower values for closer fit
    return 1 - (norm_residual / normX)

def khatri_rao(A,B):
    """
    following the outline of Kolda & Bader (2009),
    column-wise kronecker-product.
    A & B: two matricies with the same number of columns
    """
    if(A.shape[1] == B.shape[1]):
        rank = A.shape[1]
        rows = A.shape[0] * B.shape[0]
        # khatri row product of A and B will be rows x rank matrix
        P = np.zeros((rows,rank))
        for i in range(rank):
  	        # kronecker product of two vectors is the outer product
  	        # of the two
            ab = np.outer(A[:, i], B[:, i])
            # ab is vectorized before assigned to the i'th column of P
            P[:,i] = ab.flatten()
        return P
    else: 
        raise ValueError("Matricies must have the same # of columns")

def n_mode_prod(X, A, n):
    """
    Calculates the n-mode product of a tensor and matrix A
    """
    shape = list(X.get_shape())
    # check that dimensions allows for matrix multiplication
    if(shape[n] == A.shape[1]):
        Xn = unfold_tf(X, n)
        Xn = tf.matmul(A, Xn)
        # alternatively
        # Xn = np.matmul(Xn, A.T)

        # the folded tensor will have nth dimension A.shape[0]
        shape[n] = A.shape[0]
    
        return refold_tf(Xn, shape, n)

    else:
  	    raise ValueError("Xn and A does not allow for matrix multiplication")



def kruskal(G, U):
    """
    Kruskal operator, takes core matrix and list och 
    unfolded matricies.

    G: tensor 
    U: list of factor matricies

    X = G times1 U1 times2 U2 ... timesN UN
    """
    # need to keep in mind that the shape returns
    # the dimension in a completely different order in 
    # comparison with the litterature. 
    N = len(U)
    for n in range(N):
  	    # n_mode_prod does the unfolding and refolding
  	    X = n_mode_prod(G, U[n], n)
    return(X)

