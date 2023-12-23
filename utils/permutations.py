# -*- coding: utf-8 -*-
"""
Purpose
-------
Get the permutations of multiple indpendent variables.

Classes
-------
None

Functions
---------
permutations(*u_ivs)
    Creates all permutations of the independent variables.


Change log
----------     
5 May 21 - Separated these functions from a more general toolkit.

"""

import numpy as np

def permutations(*u_ivs):
    '''
    Pass in any number of unique independent variables and it will return
    lists will all possible combinations. 
    
    Usage Example:
        u_a = [1, 2]
        u_b = [3, 4]
        
        a, b = permutations(u_a, u_b)
        
        print(a) 'prints array([1, 1 , 2, 2])'
        print(b) 'prints array([3, 4 , 3, 4])'

    Parameters
    ----------
    u_ivs : array like (N lists)
        Unique values of independent variables. Each independent variable
        should be an array.

    Returns
    -------
    ivs : arrays (N lists)()
        Lists values of all permutations of the independent variables. These
        are in relation to the other independent variables. Review the output
        here because the independent variable order is inconsistent.
    '''
    
    # creates a multidimensional matrix with all of the values.
    mesh = np.array(np.meshgrid(*u_ivs))
    
    # reorganizes the meshgrid so that each column is an iv.
    mesh = mesh.T.reshape(-1, len(u_ivs))

    # seperates the columns   
    ivs = np.hsplit(mesh, len(u_ivs))
    
    # prevents each column from being an array of single values
    for c, i in enumerate(ivs):
        ivs[c] = ivs[c].reshape(-1)
        ivs[c] = ivs[c].astype(np.array(u_ivs[c]).dtype)

    return ivs