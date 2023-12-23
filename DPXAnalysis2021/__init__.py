# -*- coding: utf-8 -*-
"""
Tools for analyzing data to go with the 2021 paper.

Modules: 
    dpx - Functions to analyze DPX performance and to create reaction times.
    tunecurve - Functions to create tuning curve files.
    project_plots - Functions to recreate figures

Author: Olivia Calvin, University of Minnesota
Last updated on 3 May 2021

"""

from .dpx import (load_dpx, analyze_dpx, smrz_dpx, act_bmp_rep,
                  reconstruct_DPX_actbump, reconstruct_rep_actbump,
                  create_all_actbmp_dpx_rts, analyze_actbmp_timings, dpx_rep)
from .tunecurve import (radial_tc, create_tc_files, trial_packet_comp,
                        EI_trial_packet_comp, create_fldr_tcs_comps,
                        create_EI_fldr_tcs_comps)
from .project_plots import (cp_raster_plot, dpx_raster_plot, 
                            cross_study_err_rates, smrz_EI_survey, 
                            DPX_smry, DPX_rt_smry)