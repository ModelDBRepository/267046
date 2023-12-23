# -*- coding: utf-8 -*-
"""
Purpose
-------
Fits nonlinear functions to data.

Classes
-------
None

Functions
---------
heaviside(iv, dv)
    Fit a Heaviside function.
logistic(iv, dv)
    Fit a logistic function.

Change log
----------     
10 Feb 21 - Separated these functions from a more general toolkit.
"""

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def heaviside(iv, dv, plot=False, **kwargs):
    '''
    Fit a heaviside function.

    Parameters
    ----------
    iv : array(float)
        The independent variable's values.
    dv : array(float)
        The dependent variable's values.
    plot : bool, optional
        Whether the fit should be plotted. The default is False.
        
    Can be passed kwargs for scipy.optimize.curve_fit

    Returns
    -------
    params : list
        Optimal parameter values.
    cov : list
        The estimated covariance of the parameters.
    
    See scipy.optimize.curve_fit for more details.

    '''
    
    # Ensure the values are in numpy format
    iv = np.array(iv)
    dv = np.array(dv)
    
    # Set the bounds of the function (by parameter) if not specified.
    kwargs.setdefault('bounds', ([np.min(iv), -np.inf],
                                 [np.max(iv), np.inf]
                                 )
                      )
      
    # Fit a smoothed heaviside
    func = lambda iv, x, k : (0.5 + np.tanh(k * (iv-x))/2)    
    params, cov = curve_fit(func, iv, dv, **kwargs)
    
    if plot:
        _plot_fit(iv, dv, func, *params, label='Heaviside Fit')
        
    return params, cov

def logistic(iv, dv, plot=False, **kwargs):
    '''
    Fit a logistic function.

    Parameters
    ----------
    iv : array(float)
        The independent variable's values.
    dv : array(float)
        The dependent variable's values.
    plot : bool, optional
        Whether the fit should be plotted. The default is False.
        
    Can be passed kwargs for scipy.optimize.curve_fit

    Returns
    -------
    params : list
        Optimal parameter values.
    cov : list
        The estimated covariance of the parameters.
    
    See scipy.optimize.curve_fit for more details.

    '''
    
    # Ensure the values are in numpy format
    iv = np.array(iv)
    dv = np.array(dv)
    
    # Set the bounds of the function (by parameter) if not specified.
    diff = np.max(dv) - np.min(dv)
    d_min = np.min(dv)
    if d_min == 0: d_min += 0.000000001
    d_range = d_min * 0.1
    kwargs.setdefault('bounds', ([diff * 0.999, 
                                  -np.inf, 
                                  np.min(iv), 
                                  (d_min-d_range) if d_min >= 0 else (d_min+d_range)
                                  ],
                                 [diff * 1.001, 
                                 np.inf, 
                                 np.max(iv),
                                 (d_min+d_range) if d_min >= 0 else (d_min-d_range)
                                 ]
                                 )                                
                      )
    
    # Create and fit the logistic function
    func = lambda iv, a, b, c, d: (a / (1 + np.exp(-b * (iv - c))) + d)
    params, cov = curve_fit(func, iv, dv, **kwargs)
    
    if plot:
        _plot_fit(iv, dv, func, *params, label='Logisitic Fit')
        
    return params, cov

def _plot_fit(iv, dv, func, *params, label='Fit'):
    '''
    Plot the function fit.

    Parameters
    ----------
    iv : array(float)
        The independent variable's values.
    dv : array(float)
        The dependent variable's values.
    func : lambda
        The function used for the fit.
    *params : list(float)
        The best parameters from the fit.
    label : string, optional
        The legend's label for the fit. The default is 'Fit'.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.plot(iv, dv, label='Data')
    _l_space = np.linspace(np.min(iv), np.max(iv), 1000)
    plt.plot(_l_space, func(_l_space, *params), color='r', linestyle='--',
             label=label)
    plt.legend()

"""
def dumb_heaviside(iv, dv, plot=False, **kwargs):
    '''
    Fit a heaviside function by iterating through
    potential solutions.

    Parameters
    ----------
    iv : array(float)
        The independent variable's values.
    dv : array(float)
        The dependent variable's values.
    plot : bool, optional
        Whether the fit should be plotted. The default is False.
        
    Can be passed kwargs for scipy.optimize.curve_fit

    Returns
    -------
    params : list
        Optimal parameter values.
    cov : list
        The estimated covariance of the parameters.
    
    See scipy.optimize.curve_fit for more details.

    '''
    
    # Ensure the values are in numpy format
    iv = np.array(iv)
    dv = np.array(dv)
    
    # Set the bounds of the function (by parameter) if not specified.
    kwargs.setdefault('bounds', ([np.min(iv), -np.inf],
                                 [np.max(iv), np.inf]
                                 )
                      )
      
    # Fit a smoothed heaviside
    func = lambda iv, x, k : (0.5 + np.tanh(k * (iv-x))/2)    
    params, cov = curve_fit(func, iv, dv, **kwargs)
    
    if plot:
        _plot_fit(iv, dv, func, *params, label='Heaviside Fit')
        
    return params, cov
"""