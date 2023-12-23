# -*- coding: utf-8 -*-
"""
Purpose
-------
Tuning curve based analyses for the 2021 DPX Agent Manuscript.

Classes
-------
None

Functions
---------
radial_tc(spk_dc)
    Calculates a simulated population's tuning curve in the specific 
    scenario that they can be interchangeable and that they each have
    a radial direction assigned to them.  
create_tc_files(root_folder)
    Creates a tuning curve file.
trial_packet_comp(spk_dc, tc)
    Compares the population activity with the tuning curve for each
    stimulus. Listed parameters are for the DPX task.
EI_trial_packet_comp(spk_dc, tc)
    Compares the population activity with the tuning curve for each
    stimulus. Listed parameters are for the EI balance experiments.
create_fldr_tcs_comps(fldr)
    Creates tuning curves and comparisons within a DPX simulation folder.
create_EI_fldr_tcs_comps(fldr)
    Creates tuning curves and comparisons within an EI simulation folder.


Change log
----------     
2 May 21 - Separated these functions from a more general toolkit.

"""

import os
from bisect import bisect_left

import numpy as np

from utils import fetch_files
from collect import spikeDC


def radial_tc(spk_dc, start=1400, end=1500, 
                 time_win=25, dt=5, trials=None):
    '''
    This function calculates a simulated population's tuning curve
    in the specific scenario that they can be interchangeable
    and that they each have a radial direction assigned to them.

    Parameters
    ----------
    spk_dc : spikeDC
        The spiking activity that the tuning curves need to be
        determined from.
    start : int, optional
        The start of the window that will be used to calculate
        the tuning curve. The default is 1400 ms.
    end : int, optional
        The end of the window that will be used to calculate
        the tuning curve. The default is 1400 ms.
    time_win : int, optional
        The duration of the sliding window that is slid over
        between the start and end. The default is 25.
    dt : float, optional
        The change in time. The default is 5 ms.
    trials : list, optional
        The trials that the tuning curve should be calculated
        over. The default is None, which means that all trials
        will be used.

    Returns
    -------
    tune_curve : np.array(float)
        The calculated tuning curve.

    '''
    mapping = np.linspace(0, 2 * np.pi, num=spk_dc.size)    
    
    sum_adj_rates = np.zeros(mapping.size)
    mag = None
    vect = None
    if trials is None:
        trials = spk_dc.trials
    
    circ = np.pi*2
    half_map = int(len(mapping) / 2)

    for t in trials:
        spk_cnts = np.zeros(len(mapping))
        
        spk_dc.load_trial(t)
        print("Analyzing trial", t)
        
        for time in np.arange(start+time_win, end, step=dt): 
            vect, mag, spk_cnts = spk_dc.pop_vector(mapping, 
                                                time - time_win, 
                                                time,
                                                return_counts=True)
            # Adjusts for the vectors being from np.pi to -np.pi
            if vect < 0: vect += circ
            
            vect_index = bisect_left(mapping, vect)
            sum_adj_rates += np.roll(spk_cnts, half_map - vect_index)
    
    mean_adj_rates = sum_adj_rates / len(spk_dc.trials)
    tune_curve = mean_adj_rates / mean_adj_rates.max()

    return tune_curve

def create_tc_files(root_folder, 
                    file_name='percPyr.npz', 
                    tc_fname='percTCs.npz',
                    neurons=1024,
                    box_car_size=11,
                    **kwargs):
    '''
    Creates a tuning curve file.

    Parameters
    ----------
    root_folder : string
        The root folder that will have its subdirectories searched.
    file_name : string, optional
        The name of the files that the tuning curve needs to be calculated
        for. The default is 'percPyr.npz'.
    tc_fname : string, optional
        The name to save the tuning curves as. The default is 'percTCs.npz'.
    neurons : int, optional
        The size of the neural population. The default is 1024.
    box_car_size : int, optional
        The size of the box car smoother that needs to be applied. The 
        default is 11.
    **kwargs : dict
        Keyword arguments to pass to the radial_tc function.

    Returns
    -------
    None.

    '''
    files = fetch_files(root_folder, file_name)
    data = spikeDC(neurons)

    # Create the boxcar for smoothing
    box_car = np.ones(box_car_size) / box_car_size
        
    for f in files:
        path, file = os.path.split(f)
        
        data.load(f, 1)
        print("Loaded", f)
        print("Creating representation tuning curve.")
        tc = radial_tc(data, **kwargs)
        
        # smooth the tuning curve
        smth_tc = np.convolve(tc, box_car)
        rem_borders = int((len(smth_tc) - len(tc))/2)
        smth_tc = smth_tc[rem_borders:-rem_borders]
    
        np.savez_compressed(path+'/'+tc_fname,
                            stim=smth_tc
                            )
        print(f, "completed.")

def trial_packet_comp(spk_dc, tc,
                      stim_dict={'A':0.3, 'B':0.7, 'X':1.3, 'Y':1.7},
                      neurons=1024,
                      wind=25, start=0, end=6600, dt=1, trials=None):
    '''
    Compares the population activity with the tuning curve for each
    stimulus. Listed parameters are for the DPX task.

    Parameters
    ----------
    spk_dc : spikeDC
        The pyramidal cell population spiking.
    tc : np.array(float)
        The population tuning curve.
    stim_dict : dict, optional
        The radial directions that are associated with the stimuli. 
        The default is {'A':0.3, 'B':0.7, 'X':1.3, 'Y':1.7}.
    neurons : int, optional
        The number of neurons in the population. The default is 1024.
    wind : int, optional
        The size of the window for the packet comparison. 
        The default is 25.
    start : int, optional
        The starting time. The default is 0 ms.
    end : int, optional
        The end time. The default is 6600 ms.
    dt : float, optional
        The change in time for the tuning curve to be calculated over.
        The default is 1 ms.
    trials : list, optional
        Permits a subset of trials to be examined. The default is None,
        which means that all trials are examined.

    Returns
    -------
    return_matrix : dict
        This dictionary contains arrays for the trial, time, and the
        comparisons to each stimuli's tuning curve.

    '''
    
    if trials is None: trials = spk_dc.trials
    
    half_size = int(len(tc)/2)
    tc_matrix = np.zeros([len(tc), len(tc)])
    
    for index in range(len(tc)):
        tc_matrix[index,:] = np.roll(tc, index - half_size)
    
    b_matrix = tc_matrix / np.sum(tc_matrix, axis=1) 
    
    stim_tcs = np.zeros([len(stim_dict.keys()), len(tc)])
    for index, val in enumerate(stim_dict.values()):
        roll = int(val/2 * neurons) - half_size
        stim_tcs[index,:] = np.sum(b_matrix * np.roll(tc, roll), axis=1)
    
    output_matrix = np.zeros([int((end-start-wind)/dt) * len(trials),
                             len(stim_dict.keys())+2]
                            )
    trl_matrix = np.zeros([int((end-start-wind)/dt),
                             len(stim_dict.keys())]
                         )
    
    times = np.arange(start+wind, end, dt)
    trl_size = len(times)
    
    # Go through each trial
    for trl_index, trial in enumerate(trials):
        spk_dc.load_trial(trial)
        print('Evaluating trial', trial)
    
        # Go through each time step
        for index, time in enumerate(times):
            # Create the activity packet
            spk_cnts = spk_dc.get_spike_counts(time-wind, time)
            spks = np.max(spk_cnts)
            if spks == 0: spks = 1
            spk_cnts = spk_cnts/spks
            act_packet = np.sum(spk_cnts * b_matrix, axis=1)
            
            # Compare the packet to the ideal tuning curves
            pack_diff = stim_tcs - act_packet
            diffs = np.sum(np.square(pack_diff), axis=1)
            trl_matrix[index,:] = diffs
        
        mat_start = trl_index * trl_size
        mat_end = mat_start + trl_size
        output_matrix[mat_start:mat_end, 0] = trial
        output_matrix[mat_start:mat_end, 1] = times
        output_matrix[mat_start:mat_end, 2:] = trl_matrix
    
    return_matrix = {'trial':output_matrix[:,0].astype(int),
                     'times':output_matrix[:,1].astype(int)}
    for index, key in enumerate(stim_dict.keys()):
        return_matrix.update({key:output_matrix[:,index+2]})
    
    return return_matrix

def EI_trial_packet_comp(spk_dc, tc, **kwargs):
    '''
    Compares the population activity with the tuning curve for each
    stimulus. Listed parameters are for the EI balance experiments.

    Parameters
    ----------
    spk_dc : spikeDC
        The pyramidal cell population spiking.
    tc : np.array(float)
        The population tuning curve.
    **kwargs : dict
        Parameters to be passed trial_packet_comp.

    Returns
    -------
    return_matrix : dict
        This dictionary contains arrays for the trial, time, and the
        comparisons to each stimuli's tuning curve.

    '''
    kwargs.setdefault('stim_dict', {'C':0.5, 'P':1.5})
    kwargs.setdefault('start', 0)
    kwargs.setdefault('end', 4000)
    
    return trial_packet_comp(spk_dc, tc, **kwargs)    

def create_fldr_tcs_comps(fldr,
                          perc_f_name='percPyr.npz',
                          perc_tc_fname='percTC.npz', 
                          perc_comp_fname='percTC_comp.npz',
                          mem_f_name='memPyr.npz',
                          mem_tc_fname='memTC.npz', 
                          mem_comp_fname='memTC_comp.npz',
                          box_car_size=11):
    '''
    Creates tuning curves and comparisons within a DPX simulation folder.

    Parameters
    ----------
    fldr : string
        The folder's location and name.
    perc_f_name : string, optional
        The name of the perception ring attractor's activity file.
        The default is 'percPyr.npz'.
    perc_tc_fname : string, optional
        The name of the perception ring attractor's tuning curve file.
        The default is 'percTC.npz'.
    perc_comp_fname : string, optional
        The name of the perception ring attractor's activity comparison
        to the tuning curve. The default is 'percTC_comp.npz'.
    mem_f_name : string, optional
        The name of the memory ring attractor's activity file.
        The default is 'memPyr.npz'.
    mem_tc_fname : string, optional
        The name of the memory ring attractor's tuning curve file.
        The default is 'memTC.npz'.
    mem_comp_fname : string, optional
        The name of the memory ring attractor's activity comparison
        to the tuning curve. The default is 'memTC_comp.npz'.
    box_car_size : int, optional
        The length of the boxcar smooth. The default is 11.

    '''
    box_car = np.ones(box_car_size) / box_car_size    
    spk_dc = spikeDC(1024)
    
    # Create, smooth, and save perc tuning curve
    spk_dc.load(fldr+perc_f_name, 1)
    tc = radial_tc(spk_dc)
    smth_tc = np.convolve(tc, box_car)
    rem_borders = int((len(smth_tc) - len(tc))/2)
    smth_tc = smth_tc[rem_borders:-rem_borders]
    np.savez_compressed(fldr+perc_tc_fname, stim=smth_tc)
    
    # Create perception comparison
    save_dict = trial_packet_comp(spk_dc, smth_tc)
    np.savez_compressed(fldr+perc_comp_fname, **save_dict)
    
    # Create, smooth, and save mem tuning curve
    spk_dc.load(fldr+mem_f_name, 1)
    tc = radial_tc(spk_dc)
    smth_tc = np.convolve(tc, box_car)
    rem_borders = int((len(smth_tc) - len(tc))/2)
    smth_tc = smth_tc[rem_borders:-rem_borders]
    np.savez_compressed(fldr+mem_tc_fname, stim=smth_tc)
    
    # Create memory comparison
    save_dict = trial_packet_comp(spk_dc, smth_tc)
    np.savez_compressed(fldr+mem_comp_fname, **save_dict)
    
def create_EI_fldr_tcs_comps(fldr,
                              f_name='pyr.npz',
                              tc_fname='TC.npz', 
                              comp_fname='TC_comp.npz',
                              box_car_size=11):
    '''
    Creates tuning curves and comparisons within an EI simulation folder.
    
    Parameters
    ----------
    fldr : string
        The folder's location and name.
    f_name : string optional
        The name of the pyramidal cell population's activity.
        The default is 'pyr.npz'.
    tc_fname : string, optional
        The name of the pyramidal cell population's tuning curve.
        The default is 'TC.npz'.
    comp_fname : string, optional
        The name of the pyramidal cell population's activity comparison
        to the tuning curve. The default is 'TC_comp.npz'.
    box_car_size : int, optional
        The length of the boxcar smooth. The default is 11.

    '''
    box_car = np.ones(box_car_size) / box_car_size    
    spk_dc = spikeDC(1024)
    
    # Create, smooth, and save perc tuning curve
    spk_dc.load(fldr+f_name, 1)
    tc = radial_tc(spk_dc, start=900, end=1000)
    smth_tc = np.convolve(tc, box_car)
    rem_borders = int((len(smth_tc) - len(tc))/2)
    smth_tc = smth_tc[rem_borders:-rem_borders]
    np.savez_compressed(fldr+tc_fname, stim=smth_tc)
    
    # Create perception comparison
    save_dict = EI_trial_packet_comp(spk_dc, smth_tc)
    np.savez_compressed(fldr+comp_fname, **save_dict)