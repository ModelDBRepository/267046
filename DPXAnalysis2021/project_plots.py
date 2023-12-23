# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:21:05 2020

@author: ocalvin
"""

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# Ignore the warning, this is necessary for 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

from plotting import raster
from .dpx import analyze_actbmp_timings, analyze_dpx


# Set the default font for figures
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42


'''
----------------------------------------------------------------------      
--------------------- Experiment 1: E/I Balance ----------------------
----------------------------------------------------------------------
'''

def cp_raster_plot(spike_dat_coll, *args, **kwargs):
    """
    Creates a raster plot figure for a cue-probe trial.

    Parameters
    ----------
    spike_dat_coll : SpikeDataCollector
        A recording object that records spikes during experiments
        and can load spiking information from csv's.
        
    Try help(raster_plot) for more information on other parameters
    that can be set.
    
    NOTE: Used to create Figure 2's E-G panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
       Returns the raster plot figure, so that it may be saved or
       manipulated.

    """
    # Uses set defaults in case user wants to overwrite them.
    kwargs.setdefault('ylabel', 'Associated Direction (in radians)')
    kwargs.setdefault('yticklabels', 
                      ('0', '$\\pi$/2', '$\\pi$', '3$\\pi$/2', '2$\\pi$'))
    kwargs.setdefault('xlims', 
                      (0, 4000))
    
    # Set the default parameters for the CP trials.
    _cues = {'C' : 0.5, 'P' : 1.5}              # in radians
    _cue_colors = {'C' : 'b', 'P' : 'r'}
    _cue_w = 0.2
    _starts = (500, 3000)
    _ends = (1000, 3500)
    _map_alpha = 0.15
    _highlight_alpha = 0.30

    # Create the dictionaries.
    cue_disp_map = dict()
    cue_highlight = dict()
    _i = 0
    for cue, val in _cues.items():
        cue_disp_map.update(
            {cue: 
             {'ybottom' : int(spike_dat_coll.raster_IDs[-1] * ((val - _cue_w)/2)),
              'ytop' : int(spike_dat_coll.raster_IDs[-1] * ((val + _cue_w)/2)),
              'color' : _cue_colors.get(cue),
              'alpha' : _map_alpha}}
            )
        
        cue_highlight.update({cue: {'start' : _starts[_i],
                                     'end' : _ends[_i],
                                     'alpha' : _highlight_alpha}}
                              )  
        _i += 1
  
    return raster(spike_dat_coll, cue_disp_map, cue_highlight, 
                  *args, **kwargs)    

'''
----------------------------------------------------------------------      
------------ Experiment 2: Interneuron AMPA/NMDA Balance -------------
----------------------------------------------------------------------
'''

def smrz_EI_survey(fldrs, pyr_gs, int_gs, aff_gs):
    '''
    Creates a multipanel summary of network functioning over
    a wide range of parameter values.
    
    NOTE: Used to create Figure 3.

    Parameters
    ----------
    fldrs : list(string)
        List of the data folders that need to be described.
    pyr_gs : list(float)
        List of the pyramidal cell NMDA conductances for each file.
    int_gs : list(float)
        List of the interneuron NMDA conductances for each file.
    aff_gs : list(float)
        List of the afferent AMPA conductances for each file.

    '''
    
    cueS, probeS, cueL, probeL, jump = [], [], [], [], []
    initMed, jumpMed, cueDurMed, probeDurMed = [], [], [], []
    spiralled = []
    
    # gets all of the data timing information for the various files
    for f in fldrs:
        temp = analyze_actbmp_timings(f+'TC_comp.npz', f+'TC.npz')
        cueS.append(temp['cueStart'])
        probeS.append(temp['probeStart'])
        cueL.append(temp['cueLast'])
        probeL.append(temp['probeLast'])
        jump.append(temp['jumped'])
        initMed.append(temp['initMedian'])
        jumpMed.append(temp['jumpMedian'])
        cueDurMed.append(temp['cueDurMedian'])
        probeDurMed.append(temp['probeDurMedian'])
        spiralled.append(temp['spiralled'])
    
    # Set aside the unique afferent g values
    u_affg = np.unique(aff_gs)
    
    # Turns the collected data into numpy arrays and some of the variables 
    #    into percentiles
    cueS, probeS = np.array(cueS) * 100, np.array(probeS) * 100
    cueL, probeL = np.array(cueL) * 100, np.array(probeL) * 100
    jump = np.array(jump) * 100
    initMed, jumpMed = np.array(initMed), np.array(jumpMed)
    cueDurMed, probeDurMed = np.array(cueDurMed), np.array(probeDurMed)
    spiralled = np.array(spiralled) 
    
    # Loop through all of the afferent conditions
    for a in u_affg:
        # create the subsets for this pass
        msk = np.where(aff_gs == a)
        
        # Create functional use map
        fig, ax = plt.subplots()
        norm = plt.Normalize(0, 1)
        _red = np.where(spiralled[msk] != 1, jump[msk]/100, 1)
        _green = np.where(spiralled[msk] != 1, 0, 1) 
        _blue = np.where(spiralled[msk] != 1, ((cueL[msk] - jump[msk])/100), 1)
        _colorMap = np.vstack((_red, _green, _blue)).T
        _colorMap = _colorMap.reshape(np.unique(int_gs[msk]).size,
                                      np.unique(pyr_gs[msk]).size, -3)
        _colorMap = np.flipud(_colorMap)
        
        plt.imshow(_colorMap, alpha=0.9)
        plt.ylabel('Pyramidal NMDA g')
        plt.xlabel('Interneuron NMDA g')
        tick_labels = np.round(np.linspace(np.min(pyr_gs[msk]), 
                                           np.max(pyr_gs[msk]), num = 3),3)
        tick_labels = ['{:0.2f}'.format(x) for x in tick_labels[::-1]]
        plt.yticks(np.linspace(0,np.unique(pyr_gs[msk]).size-1, num = 3), 
                   tick_labels)
        tick_labels = np.round(np.linspace(np.min(int_gs[msk]), 
                                           np.max(int_gs[msk]), num = 3),3)
        tick_labels = ['{:0.2f}'.format(x) for x in tick_labels]
        plt.xticks(np.linspace(0,np.unique(int_gs[msk]).size-1, num = 3), 
                   tick_labels)
        plt.title('Afferent AMPA g ' + str(round(a,4)))
        
        # Creates the legend
        _rpatch = mpatches.Patch(color='red', label='Jumps')
        _bpatch = mpatches.Patch(color='blue', label='Stays')
        _npatch = mpatches.Patch(color='black', label='Fails Prior To')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), 
                   handles=[_bpatch, _rpatch, _npatch], ncol=3)
        plt.show()




'''
----------------------------------------------------------------------      
-------------- Experiment 3: DPX Model NMDA Antagonism ---------------
----------------------------------------------------------------------
'''

def dpx_raster_plot(spike_dat_coll, trial_type, *args, **kwargs):
    """
    Creates a raster plot figure for a dpx trial.

    NOTE: Used to create Figures 5A-C and Supplemental 1A-C.
    
    Parameters
    ----------
    spike_dat_coll : SpikeDataCollector
        A recording object that records spikes during experiments
        and can load spiking information from csv's.
    
    trial_type : string
        The type of trial that is being plotted. For example, a
        BX trial would be 'BX'.
        
    Try help(raster_plot) for more information on other parameters
    that can be set.

    Returns
    -------
    fig : matplotlib.figure.Figure
       Returns the raster plot figure, so that it may be saved or
       manipulated.

    """
    # Uses set defaults in case user wants to overwrite them.
    kwargs.setdefault('ylabel', 'Associated Direction (in radians)')
    kwargs.setdefault('yticklabels', 
                      ('0', '$\\pi$/2', '$\\pi$', '3$\\pi$/2', '2$\\pi$'))
    kwargs.setdefault('xlims', 
                      (0, 6600))
    
    # Set the default parameters for the CP trials.
    _cues = {'A' : 0.3, 'B' : 0.7, 'X' : 1.3, 'Y' : 1.7}     # in radians
    _cue_colors = {'A' : 'b', 'B' : 'r', 'X' : 'b', 'Y' : 'r'}
    _cue_w = 0.2
    _starts = (500, 5500)
    _ends = (1500, 6000)
    _map_alpha = 0.15
    _highlight_alpha = 0.30

    # Create the dictionaries.
    cue_disp_map = dict()
    cue_highlight = dict()
    
    for cue, val in _cues.items():
        cue_disp_map.update(
            {cue: 
             {'ybottom' : int(spike_dat_coll.raster_IDs[-1] * ((val - _cue_w)/2)),
              'ytop' : int(spike_dat_coll.raster_IDs[-1] * ((val + _cue_w)/2)),
              'color' : _cue_colors.get(cue),
              'alpha' : _map_alpha}}
            )
    
    _i = 0
    
    for letters in trial_type:
        cue_highlight.update({letters: {'start' : _starts[_i],
                                        'end' : _ends[_i],
                                        'alpha' : _highlight_alpha}}
                             )  
        _i += 1
  
    return raster(spike_dat_coll, cue_disp_map, cue_highlight, 
                    *args, **kwargs)

def DPX_smry(files, values, title='DPX Performance', 
            xlabel='Percent Change', 
            ylabel='Percent Errors', ymax= None):
    '''
    Creates summary images for the DPX performance over a set of conditions.
    
    NOTE: Used to create Figures 5D-F, 8, 10A, and Supplemental 1D-F

    Parameters
    ----------
    files : list(string)
        List of the files that need to be loaded and analyzed. These should
        be DPX files.
    values : list(float)
        Parameter values that are paired with the file and will be used to 
        plot along the x-axis.
    title : string, optional
        The string label for the plot's title. The default is 
        'DPX Performance'.
    xlabel : string, optional
        The string label for the x-axis. The default is 'Percent Reduction'.
    ylabel : string, optional
        the string label for the y-axis. The default is 'Percent Errors'.
    ymax : float, optional
        If specified, then the plot's area will be from -2 to ymax. Otherwise
        it will plot based on the observed values. The default is None.

    Returns
    -------
    The figure.

    '''
    AXvalues = []
    AYvalues = []
    BXvalues = []
    BYvalues = []
    
    for f in files:
        # Analyze the data and add values to the relevant lines
        vals = analyze_dpx(f, plot=False)
        AXvalues.append(vals['AX'] * 100)
        AYvalues.append(vals['AY'] * 100)
        BXvalues.append(vals['BX'] * 100)
        BYvalues.append(vals['BY'] * 100)
    
    # Plot the values    
    fig, ax = plt.subplots()
    ax.plot(values, AXvalues, label='AX', color='black', marker='o')
    ax.plot(values, AYvalues, label='AY', color='blue', linestyle='-.', marker='v')
    ax.plot(values, BXvalues, label='BX', color='red', linestyle='--', marker='s')
    ax.plot(values, BYvalues, label='BY', color='purple', linestyle=':', marker='*')
    
   # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not ymax is None: plt.ylim(bottom=-2, top=ymax)
    plt.title(title)
    plt.legend(loc=0)
            
    return fig

def DPX_rt_smry(files, values,
                func='median', title='DPX Performance', 
                xlabel='Percent Change', ylabel='Reaction Time (ms)', 
                cue_start=5500):
    '''
    Creates summary images for the agent's reaction times 
    over the various trial types.
    
    NOTE: Used to create Figures 6B-D and 9.

    Parameters
    ----------
    files : list(string)
        List of the files that need to be loaded and analyzed. These should
        be DPX files.
    values : list(float)
        Parameter values that are paired with the file and will be used to 
        plot along the x-axis.
    func : string
        Which function should be used to plot the reaction times. The two
        options are 'mean' and 'median'. The default is 'median'.
    title : string, optional
        The string label for the plot's title. The default is 
        'DPX Performance'.
    xlabel : string, optional
        The string label for the x-axis. The default is 'Percent Reduction'.
    ylabel : string, optional
        the string label for the y-axis. The default is 'Percent Errors'.
    cue_start: float, optional
        The offset for when the agent's response begins.

    Returns
    -------
    The figure.

    '''
    AXvalues = []
    AYvalues = []
    BXvalues = []
    BYvalues = []
    
    if func == 'median':
        fun = np.median
    elif func == 'mean':
        fun = np.mean
    
    for f in files:
        # Analyze the data and add values to the relevant lines
        df = pd.read_csv(f)
        ss = df[np.all(np.stack([(df.cue == 'A'),(df.probe == 'X')]),axis=0)]
        AXvalues.append(fun(ss.rt) - cue_start)
        ss = df[np.all(np.stack([(df.cue == 'A'),(df.probe == 'Y')]),axis=0)]
        AYvalues.append(fun(ss.rt) - cue_start)
        ss = df[np.all(np.stack([(df.cue == 'B'),(df.probe == 'X')]),axis=0)]
        BXvalues.append(fun(ss.rt) - cue_start)
        ss = df[np.all(np.stack([(df.cue == 'B'),(df.probe == 'Y')]),axis=0)]
        BYvalues.append(fun(ss.rt) - cue_start)
    
    # Plot the values    
    fig, ax = plt.subplots()
    ax.plot(values, AXvalues, label='AX', color='black', marker='o')
    ax.plot(values, AYvalues, label='AY', color='blue', linestyle='-.', marker='v')
    ax.plot(values, BXvalues, label='BX', color='red', linestyle='--', marker='s')
    ax.plot(values, BYvalues, label='BY', color='purple', linestyle=':', marker='*')
    
   # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
            
    return fig  

'''
----------------------------------------------------------------------      
----------------------------- SUPPLEMENT -----------------------------
----------------------------------------------------------------------
'''

def cross_study_err_rates(file):
    '''
    Creates a matrix of a set of cross correlations

    NOTE: Used to make Supplemental 2.

    Parameters
    ----------
    file : string
        The data file that is used for the analysis.

    Returns
    -------
    None.

    '''
    
    data = pd.read_csv(file)
    err_types = ['AX', 'BX', 'AY', 'BY']
    xlims = [-0.2, 0.4]
    ylims = [-0.11, 0.2]
    reg_range = np.linspace(xlims[0], xlims[1], 100)
    
    # Create error differences
    data['incomplete'] = np.zeros(len(data), dtype=bool)
    for err in err_types:
        data['diff'+err] = data['Scz'+err] - data['Con'+err]
        data['incomplete'] = np.logical_or(data['incomplete'],
                                           data['diff'+err].isnull()
                                           )
    
    fig, s_plots = plt.subplots(4,4)
    for ind, err in enumerate(err_types):
        y_notnan = data['diff'+err].notna()
        o_y_dat = np.array(data['diff'+err])
        
        for ind2, err2 in enumerate(err_types):
            x_notnan = data['diff'+err2].notna()
            subset = np.logical_and(x_notnan, y_notnan)
            
            x_dat = np.array(data['diff'+err2][~data['incomplete']])
            y_dat = o_y_dat[~data['incomplete']]
            
            
            # plot data points and setup axes
            sp = s_plots[ind, ind2]
            sp.scatter(x_dat, y_dat, color='black')
            x_dat = np.array(data['diff'+err2][subset])
            y_dat = o_y_dat[subset]
            sp.scatter(x_dat, y_dat, color='black', facecolors='none')            
            sp.set_xlim(xlims)
            sp.set_ylim(ylims)
            sp.axhline(y=0, c='gray', lw=0.5, zorder=0, ls='--')
            sp.axvline(x=0, c='gray', lw=0.5, zorder=0, ls='--')            
            if ind == (len(err_types)-1): sp.set_xlabel(err2)
            if ind2 == 0: sp.set_ylabel(err)
            
            # Fit regression
            model = LinearRegression().fit(x_dat.reshape(-1,1), y_dat)
            reg_ys = model.predict(reg_range.reshape(-1,1))
            sp.plot(reg_range, reg_ys, lw=1, color='black')
            r_sq = str(np.round(model.score(x_dat.reshape(-1,1), y_dat),2))
            sp.text(-0.10, 0.32, r'$R^2$ = ' + r_sq)