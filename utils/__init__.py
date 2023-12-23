# -*- coding: utf-8 -*-
"""
Convenient and reusable utility functions.

Modules: 
    fmanip - File and folder retrieval and merging.
    ivprobs - Methods for solving initial value problems.

Author: Olivia Calvin, Unversity of Minnesota
Last updated on Tue 11 Feb 2021
"""

#import fmanip, ivprobs
from .fmanip import fetch_files, sort_files_by_modtime, merge_csvs, merge_npzs
from .fmanip import merge_fldrs, merge_dir
from . import ivprobs
from .permutations import permutations


__all__ = ['fetch_files',
	   'sort_files_by_modtime',
           'merge_csvs',
           'merge_npzs',
           'merge_fldrs',
           'merge_dir',
           'permutations',
	   'ivprobs']
