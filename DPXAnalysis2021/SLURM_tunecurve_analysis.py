# -*- coding: utf-8 -*-
"""
    Call to perform a tuning curve analysis to a DPX folder.
    
"""

import argparse

from .tunecurve import create_fldr_tcs_comps

# ------------- Parse Program Arguments --------------------

parser = argparse.ArgumentParser(
                                prog='SLURM_tunecurve_analysis.py', 
                                description='Computes tuning curve analysis for a DPX experiment.', 
                                usage='%(prog)s folder [options]'
                                )

parser.add_argument(
     'folder', 
     type=str,
     help='Folder that data is in and where created files will be stored.'
     )

args = vars(parser.parse_args())
fldr = args['folder']

# -------------- Execute Task ---------------

create_fldr_tcs_comps(fldr)