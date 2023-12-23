# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:54:35 2020

@author: ocalv
"""

import random
import os
import bisect as bi

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from utils import fetch_files

def load_dpx(fname, **kwargs):
    """
    Loads DPX data from the listed location and adds a column for 
    whether the participant gave the correct answer.

    Parameters
    ----------
    fname : string
        The name and pathway of the file.
    
    Returns
    -------
    dat : numpy.ndarray(N,)
        A data structure that has N entries and six columns: 'trial', 
        'cue', 'probe', 'correct', 'action', and 'right'. They have 
        the following meanings:
            
            trial : The trial number.
            cue : The cue that was presented.
            probe : The probe that was presented.
            correct : What the correct response is.
            action : What action the participant took.
            acc : Whether the action taken was correct.

    """
    # The wrapper converts fname into a data file. This copy is here
    #   simply to improve readability
    dat = pd.read_csv(fname)
            
    # Handles formatting differences with older files
    if 'acc' not in dat.columns:
        dat = dat.assign(acc=np.zeros(len(dat), dtype=bool))
    if '# trial' in dat.columns:
        dat = dat.rename(columns={'# trial': 'trial'})

    # Adds field to the data for whether the agent made the correct action.
    dat = dat.assign(acc=dat['correct'] == dat['action'])

    return dat

'''
----------------------------------------------------------------------
--------------------------Summarize Behavior -------------------------
----------------------------------------------------------------------
'''

def analyze_dpx(fname):
    """
    Analyzes choice behavior on the DPX.

    Parameters
    ----------
    fname : string
        The name and pathway of the file. Just the name of the 
        of the file needs to be passed if the pathway is also
        passed via the pathway parameter (a kwarg).

    Returns
    -------
    rv : dictionary
        A dictionary of the proportion of values that are incorrect for
        each trial type ('AX', 'AY', 'BX' & 'BY').

    """
    dat = load_dpx(fname)
    
    # Sets up the return value dictionary
    rv = {'AX': 0,'AY': 0,'BX': 0,'BY': 0}
      
    # Proportion correct
    for c_p, val in rv.items():
        mask = (np.where(dat['cue'] == c_p[0], True, False)
              & np.where(dat['probe'] == c_p[1], True, False))        
        rv[c_p] = ((len(dat.iloc[mask]) - np.sum(dat.iloc[mask]['acc'])) 
                    / len(dat.iloc[mask])
                    )

    return rv

def smrz_dpx(dpx_dat, st_time=5500, mx_time=6600, title=None, bins=20):
    '''
    Create a quick summary of DPX performance and reaction times.

    Parameters
    ----------
    dpx_dat : string
        Location of the DPX file.
    st_time : int, optional
        When actions can start to occur. The default is 5500 (ms).
    mx_time : int, optional
        When actions stop occurring. The default is 6600 (ms).
    title : string, optional
        Title for the summary plots. The default is None.
        
    '''
    dpx_dat = load_dpx(dpx_dat)
    
    # Assign the subsets (i.e., AX, AY, BX, BY)
    ss_ax = dpx_dat[dpx_dat['cue'] == 'A']
    ss_ay = ss_ax[ss_ax['probe'] == 'Y']
    ss_ax = ss_ax[ss_ax['probe'] == 'X']
    ss_bx = dpx_dat[dpx_dat['cue'] == 'B']
    ss_by = ss_bx[ss_bx['probe'] == 'Y']
    ss_bx = ss_bx[ss_bx['probe'] == 'X']
    
    # Plot histogram of response times
    plt.figure()
    plt.hist(ss_ax.rt - st_time, bins=bins, histtype='step', label='AX')
    plt.hist(ss_ay.rt - st_time, bins=bins, histtype='step', label='AY')
    plt.hist(ss_bx.rt - st_time, bins=bins, histtype='step', label='BX')
    plt.hist(ss_by.rt - st_time, bins=bins, histtype='step', label='BY')
    plt.xlabel('Reaction Time (ms)')
    plt.ylabel('Count')
    plt.xlim(0, mx_time - st_time)
    plt.title(title)
    plt.legend()
    
    # Plot Error Bars
    fig, ax = plt.subplots()
    error_rates = (1.005 - np.array([np.sum(ss_ax.acc)/len(ss_ax.acc),
                                    np.sum(ss_ay.acc)/len(ss_ay.acc),
                                    np.sum(ss_bx.acc)/len(ss_bx.acc),
                                    np.sum(ss_by.acc)/len(ss_by.acc)
                                    ])
                   ) * 100
    pos = np.arange(len(error_rates))
    ax.bar(pos, error_rates, width=0.5, tick_label=['AX','AY','BX','BY'])
    plt.ylim(-0.05,100)
    plt.ylabel('Error Rates')
    plt.title(title)

'''
----------------------------------------------------------------------
---------------------- Response Probabilities ------------------------
----------------------------------------------------------------------
'''

def act_bmp_rep(tc_comp, neurons=1024, step_size=1, thres=0.75):
    ig_cols = ['trial', 'times', 'null']
    
    # Set up the time values that the response probabilities will be
    #    calculated for
    start_time = int(np.min(tc_comp.f.times))
    end_time = int(np.max(tc_comp.f.times))
    times = np.arange(start_time, end_time, step=step_size)

    # Greedy select representation and bringing it into context
    rep = None
    for f in tc_comp.files:
        if f not in ig_cols:
            if rep is not None:
                rep = np.append(rep, tc_comp[f].reshape(-1,1), axis=1)
            else:
                rep = tc_comp[f].reshape(-1,1)
    
    rep = rep / ((np.sum(rep, axis=1).reshape(-1,1) - rep) / (rep.shape[1]-1))
    rep = np.where(rep > thres, 1, rep)
    rep = 1 - rep

    return times, rep

def reconstruct_rep_actbump(perc_tc_comp, mem_tc_comp, 
                            perc_w={'X': 0.25, 'Y': 2.00},
                            mem_w={'A': 1.00, 'B': 1.25},
                            acc_tau=80, softmax_tau=15,
                            consist_rep=50, **kwargs
                            ):
    times, perc_rep = act_bmp_rep(perc_tc_comp, **kwargs)
    times, mem_rep = act_bmp_rep(mem_tc_comp, **kwargs)
    
    #  Set up the trial matrixes that will be filled
    trials = np.unique(perc_tc_comp.f.trial)
    probe_rep = np.zeros(len(trials))
    resp_probs = np.zeros([len(trials), len(times)])   
    # guess_eff = np.ones(empty_row.shape)
    # guess_eff = guess_adj * np.convolve(guess_eff, dcy_eff)[:end_time]
    act_left = np.zeros(resp_probs.shape) #+ guess_eff
    act_right = np.zeros(resp_probs.shape)
    
    # Effect of decay  
    dcy_eff = np.exp(-np.arange(1000)/acc_tau)
    
    for index, trial in enumerate(trials):       
        # Determine the subset indices
        p_st = bi.bisect_left(perc_tc_comp.f.trial, trial) + 1
        p_end = bi.bisect_right(perc_tc_comp.f.trial, trial)
        m_st = bi.bisect_left(mem_tc_comp.f.trial, trial) + 1
        m_end = bi.bisect_right(mem_tc_comp.f.trial, trial)
        
        # Calculate the action biases
        eff = perc_rep[p_st:p_end, 2] # X
        act_left[index] += perc_w['X'] * np.convolve(eff, dcy_eff)[:len(eff)]
        eff = perc_rep[p_st:p_end, 3] # Y
        act_right[index] += perc_w['Y'] * np.convolve(eff, dcy_eff)[:len(eff)]
        eff = mem_rep[m_st:m_end, 0] # A
        act_left[index] += mem_w['A'] * np.convolve(eff, dcy_eff)[:len(eff)]
        eff = mem_rep[m_st:m_end, 1] # B
        act_right[index] += mem_w['B'] * np.convolve(eff, dcy_eff)[:len(eff)]
        
        # Determine when the probe is represented within perception
        consist_x = np.where(perc_rep[p_st:p_end, 2] > 0, 1, 0)
        consist_x = np.convolve(consist_x, np.ones(consist_rep))[:len(consist_x)]
        x_start = np.where(consist_x == consist_rep)[0]
        consist_y = np.where(perc_rep[p_st:p_end, 3] > 0, 1, 0)
        consist_y = np.convolve(consist_y, np.ones(consist_rep))[:len(consist_y)]
        y_start = np.where(consist_y == consist_rep)[0]
        if len(x_start) > 0:
            x_start = x_start[0] - consist_rep
        else:
            x_start = 0
        if len(y_start) > 0:
            y_start = y_start[0] - consist_rep
        else:
            y_start = 0
        probe_rep[index] = max(x_start, y_start)
        print("Completed trial", trial)
    
    act_left = np.exp(act_left / softmax_tau)
    act_right = np.exp(act_right / softmax_tau)
    resp_probs = act_left / (act_left + act_right)
            
    return resp_probs, probe_rep

'''
----------------------------------------------------------------------
-------------------------- Reconstruct DPX ---------------------------
----------------------------------------------------------------------
'''
                          
def create_all_actbmp_dpx_rts(srcdir, 
                               perc_tc_comp_fn='percTC_comp.npz', 
                               mem_tc_comp_fn='memTC_comp.npz', 
                               perc_tc_fn='percTC.npz', 
                               mem_tc_fn='memTC.npz', 
                               dpx_fn='dpx.csv', 
                               new_fn='dpx_ab.csv',
                               **kwargs):
    
    # Retrieve all of the folders based on folders with spike times
    files = fetch_files(srcdir, dpx_fn)
    dirs = [os.path.dirname(f) + '/' for f in files]

    for d in dirs:
        print("Calculating rts for", d)
        reconstruct_DPX_actbump(d + dpx_fn, 
                                d + perc_tc_comp_fn, d + mem_tc_comp_fn, 
                                d + perc_tc_fn, d + mem_tc_fn, 
                                new_file_name=new_fn,
                                **kwargs)



def reconstruct_DPX_actbump(dpx_file, perc_TC_comp, mem_TC_comp,
                            perc_TC, mem_TC,
                            end_end=6600, rt_offset=500,
                            start_thres=1.15, start_thres_sd=0.075,
                            thres_slope_sd=0.45, neurons=1024,
                            new_file_name=None, act_bump_pad=25,
                            **kwargs):
    # Read the relevant files
    perc_dat = np.load(perc_TC_comp)
    perc_tc = np.load(perc_TC)
    mem_dat = np.load(mem_TC_comp)   
    mem_tc = np.load(mem_TC)
    dpx = pd.read_csv(dpx_file)
    print("Loaded", dpx_file)

    
    # Handles formatting differences with older files
    if 'rt' not in dpx.columns:
        dpx = dpx.assign(rt=np.zeros(len(dpx), dtype=int))
    if '# trial' in dpx.columns:
        dpx = dpx.rename(columns={'# trial': 'trial'})
    
    # Calculates the exponent that results in ts_max_prob at 0.5.
    rp_func = lambda dur, slope, inter: (np.arange(dur) * slope + inter)
        
    # Reconstuct the response probabilities over time    
    r_probs, start_at = reconstruct_rep_actbump(perc_dat, mem_dat, **kwargs)
    
    # Create start times whenever there was no probe representation.
    gen_starts = np.where(start_at < 100)[0]
    if len(gen_starts) == len(start_at):
        print(start_at)
    for g in gen_starts:
        start_at[g] = np.percentile(np.delete(start_at, gen_starts), 
                                    np.random.rand()*100
                                    )
    start_at = start_at.astype(int)
    
    # Create the response and reaction time for each trial
    for index, row in dpx.iterrows():
        rt = end_end
        loop_cnt = 0
        while rt == end_end:
            # Determine the collapsing thresholds
            intercept = random.gauss(start_thres, start_thres_sd)
            slope = -intercept / rt_offset
            slope *= (1 + random.gauss(0, thres_slope_sd))
            left_thres = rp_func(end_end - start_at[index]- act_bump_pad,
                                 slope, 
                                 intercept
                                 )
            intercept = random.gauss(start_thres, start_thres_sd)
            intercept = -(intercept - 1)
            slope = -(intercept-1) / rt_offset
            slope *= (1 + random.gauss(0, thres_slope_sd))
            right_thres = rp_func(end_end - start_at[index] - act_bump_pad,
                                       slope, 
                                       intercept
                                       )
            
            # Determine the response time
            ss_r_probs = r_probs[index,(start_at[index]-1):]
            left_rt = np.where(ss_r_probs > left_thres)[0]
            if len(left_rt) < 1: 
                left_rt = end_end
            else:
                left_rt = left_rt[0] + start_at[index]
                
            right_rt = np.where(ss_r_probs < right_thres)[0]
            if len(right_rt) < 1: 
                right_rt = end_end
            else:
                right_rt = right_rt[0] + start_at[index]
                
            if left_rt < right_rt:
                dpx.at[index, 'rt'] = left_rt
                dpx.at[index, 'action'] = 'L'
            else:
                dpx.at[index, 'rt'] = right_rt
                dpx.at[index, 'action'] = 'R' 
            
            rt = dpx.iloc[index].rt
            
            loop_cnt += 1
            
            if loop_cnt == 100:
                # draw a new start time from the distribution
                start_at[index] = np.percentile(np.delete(start_at, index), 
                                    np.random.rand()*100
                                    )
                
            print(row['trial'],
                  "- Type:", row['cue'] + row['probe'], 
                  "Action:", row['action'], 
                  "RTs (L|R):", left_rt, "|", right_rt)

    # Save the new DPX behaviors        
    if new_file_name is not None:
        dpx_file = os.path.dirname(dpx_file) + '/' + new_file_name
    dpx.to_csv(dpx_file, index=False)
    
    perc_dat.close()
    perc_tc.close()
    mem_dat.close()
    mem_tc.close()

'''
----------------------------------------------------------------------
------------------------ Error Categorization ------------------------
----------------------------------------------------------------------
'''

def analyze_actbmp_timings(tc_comp_f, #tc_f, 
                           duration = 3999, 
                           cue_start=500, cue_end=1000, 
                           probe_start=3000, probe_end=3500, 
                           consist_rep=50, **kwargs):
    tc_comp = np.load(tc_comp_f)
    #tc = np.load(tc_f)
    times, rep = act_bmp_rep(tc_comp, **kwargs)
    
    #  Set up the trial matrixes that will be filled
    trials = np.unique(tc_comp.f.trial)
    c_start = np.zeros(len(trials))
    c_end, p_start, p_end = np.copy(c_start), np.copy(c_start), np.copy(c_start)
    pad_adj = duration - len(times)

    # Create the timing information for evaluating the start and ends of representation
    for index, trial in enumerate(trials):       
        # Determine the subset indices
        st = bi.bisect_left(tc_comp.f.trial, trial) + 1
        end = bi.bisect_right(tc_comp.f.trial, trial)
        
        # Determine when the probe is represented within perception
        cue_rep = np.where(rep[st:end, 0] > 0, 1, 0)
        cue_rep = np.convolve(cue_rep, np.ones(consist_rep))[:len(cue_rep)]
        cue_rep = np.where(cue_rep == consist_rep, 1, 0)
        probe_rep = np.where(rep[st:end, 1] > 0, 1, 0)
        probe_rep = np.convolve(probe_rep, np.ones(consist_rep))[:len(probe_rep)]
        probe_rep = np.where(probe_rep == consist_rep, 1, 0)
        
        # returns the first value
        c_start[index] = np.argmax(cue_rep > 0)
        if c_start[index] > 0:
            c_end[index] = times[-1] - np.argmax(np.flip(cue_rep) > 0) + 1
            c_start[index] += pad_adj
            #c_end[index] += pad_adj
        else:
            c_start[index] = 0
            c_end[index] = duration
        p_start[index] = np.argmax(probe_rep > 0)
        if p_start[index] > 0:
            p_end[index] = times[-1] - np.argmax(np.flip(probe_rep) > 0) + 1
            p_start[index] += pad_adj
            #p_end[index] += pad_adj
        else:
            p_start[index] = 0
            p_end[index] = duration      
    
    print('Cue', np.median(c_start), '-', np.median(c_end))
    print('Probe', np.median(p_start), '-', np.median(p_end))

    # Binarizes the timing measures into descriptions of their functioning
    cueStarted = np.where(c_start > 0, True, False)
    probeStarted = np.where(p_start > 0, True, False)
    cueLasted = cueStarted & np.where(c_end > probe_start, True, False)
    probeLasted = probeStarted & np.where(p_end >= duration, True, False)
    jumped = probeStarted & cueLasted

    # Calculates the duration measures that are based around the cue
    if np.sum(cueStarted) > 0:
        initMed = c_start - cue_start
        initMed = np.where(initMed < 0, np.nan, initMed)
        initMed = np.nanmedian(initMed)        
        cueDurMed = c_end - cue_end
        cueDurMed = np.where(cueDurMed < 0, np.nan, cueDurMed)
        cueDurMed = np.nanmedian(cueDurMed)
    else: 
        initMed = np.nan
        cueDurMed = np.nan
    
    # Calculates the duration measures that are based around the probe
    if np.sum(probeStarted) > 0:
        jumpMed = np.where(cueLasted, p_start, -1) - probe_start
        jumpMed = np.where(jumpMed < 0, np.nan, jumpMed)
        jumpMed = np.nanmedian(jumpMed)
        probeDurMed = p_end - probe_end
        probeDurMed = np.where(probeDurMed < 0, np.nan, probeDurMed)
        probeDurMed = np.nanmedian(probeDurMed)
    else:
        jumpMed = np.nan
        probeDurMed = np.nan
    
    # Turn these into percentages of all trials
    cueStarted = np.sum(cueStarted) / len(trials)
    probeStarted = np.sum(probeStarted) / len(trials)
    cueLasted = np.sum(cueLasted) / len(trials)
    probeLasted = np.sum(probeLasted) / len(trials)
    jumped = np.sum(jumped) / len(trials)
    if len(trials) < 10:
        spiralled = 1
    else:
        spiralled = 0
    
    return {'cueStart': cueStarted, 'probeStart': probeStarted, 
            'cueLast': cueLasted, 'probeLast': probeLasted, 'jumped': jumped,
            'initMedian': initMed, 'jumpMedian': jumpMed,
            'cueDurMedian': cueDurMed, 'probeDurMedian': probeDurMed,
            'spiralled': spiralled}

def dpx_rep(perc_tc_comp, mem_tc_comp,
                    cue_off=5500, probe_off=5900, **kwargs):
    perc = np.load(perc_tc_comp)
    mem = np.load(mem_tc_comp)
    
    times, perc_rep = act_bmp_rep(perc, **kwargs)
    times, mem_rep = act_bmp_rep(mem, **kwargs)
    
    trials = np.unique(perc.f.trial)    
    cue_rep = np.zeros(len(trials))
    probe_rep = np.zeros(len(trials))
    
    for index, trial in enumerate(trials):       
        # Determine the subset indices
        p_st = bi.bisect_left(perc.f.trial, trial) + 1
        m_st = bi.bisect_left(mem.f.trial, trial) + 1
        a_rep = mem_rep[m_st+cue_off, 0] > 0
        b_rep = mem_rep[m_st+cue_off, 1] > 0
        x_rep = perc_rep[p_st+probe_off, 2] > 0
        y_rep = perc_rep[p_st+probe_off, 3] > 0
        cue_rep[index] = a_rep + b_rep
        probe_rep[index] = x_rep + y_rep
        
        print('Trial', trial, 'completed.')
            
    return cue_rep, probe_rep    