# -*- coding: utf-8 -*-
"""
Tools for analyzing data.

Modules: 
    spikeDC - Plots that use the spikeDC data collection class.

Author: Olivia Calvin, Unversity of Minnesota
Last updated on Tue 11 Feb 2021
"""

from .spikePlots import raster, rate_temp, pop_vector

__all__ = ['raster',
           'rate_temp',
           'pop_vector']
