# -*- coding: utf-8 -*-
"""
Tools for analyzing data.

Modules: 
    nonlinearfit - Fit nonlinear functions.
    smooth - Smooth data.

Author: Olivia Calvin, Unversity of Minnesota
Last updated on Tue 11 Feb 2021
"""

from .nonLinFit import heaviside, logistic
from . import smooth

__all__ = ['heaviside',
           'logistic']
