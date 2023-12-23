# -*- coding: utf-8 -*-
"""
Purpose
-------
Methods for solving initial value problems.

Classes
-------
None

Functions
---------
RK2(fun, t0, y0, dt)
    Second-order Runge-Kutta algorithm (Heun's method). It can be 
    passed the derivative function, the current time, the initial 
    value, and how long it should be evaluated over. 
RK4(fun, t0, y0, dt)
    Fourth-order Runge-Kutta algorithm. It can be passed the derivative 
    function, the current time, the initial value, and how long it
    should be evaluated over.

    
26 Aug 19 - Extracted the Runge-Kutta code and added documentation. 
            This code is based on a few different sources that I 
            found online. I modified it to work a little closer to 
            SciPy's in that it can be passed a derivative function 
            that accepts the current time and the current position.

18 Sep 20 - Updated the documentation.

12 Feb 21 - Moved this code to the utils folder.

"""


def RK2(fun, t0, y0, dt):
    '''
    This is the second-order Runge-Kutta algorithm, which uses Heun's 
    method. It can be passed the derivative function, the current time,
    the initial value, and how much time it should be evaluated over.
    
    Parameters
    ----------
    fun : function
        The derivative function should accept time and the y-value
    t0 : float
        The initial time
    y0 : float
        The initial y-value
    dt : float
        The change in time.

    Returns
    -------
    predicted
        The predicted value

    '''
    k1 = fun(t0,y0)
    k2 = fun(t0 + dt, y0 + k1 * dt)

    return dt * (k1 + k2)/2

def RK4(fun, t0, y0, dt):
    '''
    This is the fourth-order Runge-Kutta algorithm. It can be passed 
    the derivative function, the current time, the initial value, and 
    how much time it should be evaluated over.
    
    Parameters
    ----------
    fun : function
        The derivative function should accept time and the y-value
    t0 : float
        The initial time
    y0 : float
        The initial y-value
    dt : float
        The change in time.

    Returns
    -------
    predicted
        The predicted value

    '''
    k1 = fun(t0,y0)
    k2 = fun(t0 + 0.5*dt, y0 + 0.5 * k1 * dt)
    k3 = fun(t0 + 0.5*dt, y0 + 0.5 * k2 * dt)
    k4 = fun(t0 + dt, y0 + k3 * dt)
    
    return dt * (k1 + 2 * k2 + 2 * k3 + k4)/6