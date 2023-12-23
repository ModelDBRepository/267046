# -*- coding: utf-8 -*-
"""
Purpose
-------
Smooth data.

Classes
-------
None

Functions
---------
spline_1D(iv, dv)
    Smooth 1-dimensional data with a univariate spline.

Change log
----------     
10 Feb 21 - Separated these functions from a more general toolkit.
"""

import numpy as np
from scipy import interpolate

def spline_1D(iv, dv, 
              smooth_fact=None, points=200, ret_spline=False,
              **kwargs):
    '''
    Smooth 1-dimensional data with a univariate spline.

    Parameters
    ----------
    iv : list(float)
        The independent variable's values.
    dv : list(float)
        The dependent variable's values.
    smooth_fact : float
        The degree to which the spline should be smoothed. The default
        is None, which means that it will use the interpolation's default.
    points : int, optional
        The number of datapoints to be interpolated. The default is 200.
    ret_spline : bool, optional
        Whether the spine should be returned. The default is False.

    Can be passed kwargs for scipy.interpolate.UnivariateSpline
    
    Returns
    -------
    iv_smooth : list(float)
        Interpolated independent variable values.
    dv_smooth : list(float)
        Interpolated dependent variable values.
    
    Optionally Returns
    ------------------
    spline : scipy.interpolate.UnivariateSpline
        the spline that was used to interpolate.
        
    See details of scipy.interpolate.UnivariateSpline for more information.

    '''
    
    # Ensures that that passed variables are in numpy format.
    _iv = np.array(iv)
    _dv = np.array(dv)
    
    # Creates a range that will be smoothed over.
    iv_smooth = np.linspace(np.min(_iv), np.max(_iv), points)
    
    # Creates and fits the spline
    spline = interpolate.UnivariateSpline(_iv, _dv, **kwargs)
    if smooth_fact is not None:
        spline.set_smoothing_factor(smooth_fact)
    dv_smooth = spline(iv_smooth)
    
    if not ret_spline:
        return iv_smooth, dv_smooth
    else:
        return spline

def boxcar_1D(values, length=None, **kwargs):
    '''
    Smooth 1-dimensional data with a univariate spline.

    Parameters
    ----------
    values : list(float)
        The values that need to be smoothed.
    length : int
        The length of the boxcar
    
    Returns
    -------
    values_smooth : list(float)
        Smoothed values.

    '''
    
    # Ensures that that passed variables are in numpy format.
    _values = np.array(values)
    
    if type(length) != int:
        raise TypeError('The length parameter must be an integer')
    
    box_car = np.ones(length) / length
    
    smth_vals = np.convolve(_values, box_car)
    borders = int((len(smth_vals) - len(_values))/2)
    smth_vals = smth_vals[borders:-borders]
    
    return smth_vals
