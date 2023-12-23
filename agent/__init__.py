# -*- coding: utf-8 -*-
"""
Tools for analyzing data.

Modules: 
    dpxAgent - Agents designed to perform on the DPX task.
    LIF_Pop - Leaky-integrate and fire populations.
    currents - Classes that controls current sources for neural populations.
    kinetics - Classes that controls the neural population's kinetics.

Author: Olivia Calvin, University of Minnesota
Last updated on Tue 11 Feb 2021
"""

#import agentBase, dpxAgent, currents, kinetics, LIF_Pop
from . import LIF_Pop, currents, kinetics
from .dpxAgent import DPX_Agent, CueProbe_Agent

__all__ = ['DPX_Agent',
           'CueProbe_Agent',
           'LIF_Pop',
           'currents',
           'kinetics'
           ]
