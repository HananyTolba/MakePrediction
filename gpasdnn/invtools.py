#!/usr/bin/env python
# -*- coding: utf-8 -*-


# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process by Deep Learning Project"
# __credits__ = ["Hanany Tolba"]
# __license__ = "Apache License 2.0"
# __version__ = "0.0.1"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hanany100@gmail.com"
# __status__ = "Production"






from scipy.linalg import lapack
import numpy as np
from numpy.testing import assert_almost_equal

inds_cache = {}

def uppertriangular_2_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=np.bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_pd_inverse(m: 'numpy array') -> 'numpy array': 
    '''
    This method calculates the inverse of a A real symmetric positive definite (n × n)-matrix
    It is much faster than Numpy's "np.linalg.inv" method for example.
    '''
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
        #print("cas 1")
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
        #print("cas 2")


    uppertriangular_2_symmetric(inv)
    return inv


def inv_col_add_update(A: 'numpy array', x: 'numpy vector', r: float) -> 'numpy array':
    '''
    This method update the inverse of a matrix appending one column and one row.
    Assume we have a  kernel matrix (A) and we known its inverse. Now,
    for prediction reason in GPR model, we expand A with one coulmn and one row, A_augmented = [A x;x.T r]
    and wish to know the inverse of A_augmented. This function calculate the inverse of
    A_augmented using block matrix inverse formular,
    hence much faster than direct inverse using for example
    Numpy function "np.linalg.inv(A_augmented)".
    '''
    x = x.reshape(-1, 1)
    #x.T = x.reshape(1, -1)

    (n, m) = A.shape
    if n != m:
        raise('Matrix should be square.')
    
    # if (A,A.T,decimal=7)
    #     raise('Matrix should be symmetric.')


    Ax = np.dot(A,x)

    q = 1 / (r - np.dot(Ax.T, x))


    M = np.block([[A + np.dot(q*Ax,Ax.T), -q*Ax], [-q * Ax.T, q]])

    
    return M

def inv_col_pop_update(A: 'numpy array',c: int) -> 'numpy array':
    '''
    This method update the inverse of a matrix  when the i-th row and column are removed.

    '''
    (n, m) = A.shape
    if n != m:
        raise('Matrix should be square.')
    

    q = A[c,c]           
    Ax=np.delete(A,c,axis=0)[:,c] 
    Ax = Ax.reshape(-1,1)            
    yA=np.delete(A,c,axis=1)[c,:]
    yA = yA.reshape(1,-1)

    M=np.delete(np.delete(A,c,axis=1),c,axis=0) - (Ax/q)@yA
    
    return M
