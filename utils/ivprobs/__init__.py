# -*- coding: utf-8 -*-
"""
Functions for initial value problems.

Functions: 
    RK2 - 2nd-order Runga-Kutta
    RK4 - 4th-order Runga-Kutta

Author: Olivia Calvin, Unversity of Minnesota
Last updated on Tue 11 Feb 2021
"""

from .rungeKutta import RK2, RK4

__all__ = ['RK2',
           'RK4']
