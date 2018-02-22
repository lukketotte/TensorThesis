import numpy as np
from utils_np import *
# from numpy.linalg import pinv
# from numpy.linalg import svd
# from sklearn.preprocessing import normalize

def parafac(tensor, rank, n_iter_max=100, init='random', tol=10e-5,
            random_state=None, verbose=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
        Computes a rank-`rank` decomposition of `tensor` [1]_ such that:
        ``tensor = [| factors[0], ..., factors[-1] |]``
    Parameters
    ----------
    tensor : ndarray
    rank  : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity
    Returns
    -------
    factors : ndarray list
            list of factors of the CP decomposition
            element `i` is of shape (tensor.shape[i], rank)
    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    # rng = check_random_state(random_state)

    ndim = len(tensor.shape)
    factors = [None] * ndim

    if init is 'random':
        for mode in range(ndim):
            init_val = np.random.uniform(low = 0, high = 1,
                                         size = tensor.shape[mode] * rank)

            factors[mode] = init_val.reshape(tensor.shape[mode], rank)

    
    rec_errors = []
    norm_tensor = norm(tensor,2)

    for iteration in range(n_iter_max):
        for mode in range(ndim):
            pseudo_inverse = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse[:] = pseudo_inverse * np.dot(np.transpose(factor), factor)
            factor = np.dot(unfold_np(tensor, mode), khatri_rao_tl(factors, mode))
            # print(pseudo_inverse)
            # print(factor)
            factor = np.transpose(np.linalg.solve(np.transpose(pseudo_inverse), np.transpose(factor)))
            factors[mode] = factor

        #if verbose or tol:
        rec_error = norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:        
                print('converged in {} iterations.'.format(iteration))
                break

    return factors