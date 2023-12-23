# -*- coding: utf-8 -*-
"""
Purpose
-------
Neuro specific plotting.

Classes
-------
None

Functions
---------
raster(spike_dat)
    Raster plot of spiking activity during a trial.
rate_temp(spike_dat)
    Creates a temperature plot of the spiking activity during a trial.
pop_vector(spike_dat)
    Plots the vector of the population activity.

Change log
----------     
10 Feb 21 - Separated these functions from a more general toolkit.
"""

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection as linCollect
import numpy as np

def raster(spike_dat, cue_disp_map=None, cue_highlight=None,
           xlabel='Time (in ms)', ylabel='Neuron', 
           xlims=None, ylims=None, yticklabels=None, cue_size=34,
           **kwargs):    
    
    """
    Function that creates a raster plot of the spikes that are 
    recorded in the spikeDC spike_dat.    

    Parameters
    ----------
    spike_dat : spikeDC
        A recording object that records spikes during experiments
        and can load spiking information from csv's.
        
    cue_disp_map : dict, optional
        A dictionary that indicates which areas of the raster will
        be highlighted, colored and labeled. The default is None.
        The format for items in this dictionary is:
            
        '[CueName]' : {'ybottom': [lowestID], 'ytop':[topID], 
                      'color': [color], 'alpha': [transarency]}        
            
    cue_highlight : dict, optional
        A dictionary that indicates which areas of the raster will
        be highlighted and colored. The default is None.
        The format for items in this dictionary is:
                   
        '[CueName]' : {'start': [start on x-axis], 'end': [end point], 
                      'alpha': [transarency]}
        
        This will load values about the cues position from
        cue_disp_map.   
        
    xlabel : string, optional
        Label of the plot's x-axis. The default is 'Time (in ms)'.
        
    ylabel : string, optional
        Label of the plot's y-axis. The default is 'Neuron'.

    xlims : tuple of float, optional
        Upper and lower limit of the x-axis. The default is None, 
        which means that it will default to the plots default 
        behavior.
        
    ylims : tuple of float, optional
        Upper and lower limit of the y-axis. The default is None, 
        which means that it will default to the plots default 
        behavior.
        
    yticks : list, optional
        Item labels for the y-axis. The default is None, which means
        that it will default to the plots default behavior.
    
    cue_size : integer, optional
        The size of the cue labels when they are written on the
        raster plot. The default is size 34.

    Returns
    -------
    fig : matplotlib.figure.Figure
       The raster plot figure
    """
    
    neurons, times = spike_dat.get_spikes()
    
    # Sets default behavior of finding the limits and does some basic
    #  verification.
    if xlims is None: 
        xlims = (0, times[-1])
    elif not isinstance(xlims, tuple): 
        raise ValueError("xlims is not a tuple.")
    if ylims is None: 
        ylims = (spike_dat.raster_IDs[0], spike_dat.raster_IDs[-1])
    elif not isinstance(ylims, tuple): 
        raise ValueError("ylims is not a tuple.")
        
    # Set kwargs defaults
    kwargs.setdefault('color', 'black')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('s', 1)

    # Creates a new figure
    fig, ax = plt.subplots()
    
    # Parameter to vertically adjust the letters
    _adj = 0.01

    # Plot the rastergram
    plt.scatter(times, neurons, **kwargs)
    
    # Highlight the regions associated with the cues and add labels
    if cue_disp_map is not None:
        for cue, d in cue_disp_map.items():
            try:
                plt.axhspan(d['ybottom'], d['ytop'], facecolor=d['color'], 
                            alpha=d['alpha'], zorder=-100
                            )
                
                _txtpos = (d['ybottom'] + d['ytop'])/2
                
                plt.text(1.025, _txtpos/spike_dat.raster_IDs[-1] - _adj,
                         cue, fontsize=cue_size, color=d['color'], 
                         transform=ax.transAxes, verticalalignment='center'
                         )
            except:
                raise ValueError(
                    "There was a problem with cue_disp_map. The cue {} had"
                    " a bad value".format(cue)
                    )
                
    # Highlight the regions associated with the cues and add labels
    if cue_highlight is not None:
        for cue, d in cue_highlight.items():
            map_item = cue_disp_map.get(cue)
            
            if map_item is None:
                raise ValueError(
                    "The cue to be highlighted ({}) does not exist in"
                    " cue_disp_map.".format(cue)
                    )
            
            try:
                plt.axhspan(map_item['ybottom'], map_item['ytop'],
                            d['start'] / times[-1], 
                            d['end'] / times[-1],
                            facecolor=map_item['color'], 
                            alpha=d['alpha'], zorder=-99
                            )
            except:
                raise ValueError(
                    "There was a problem with cue_highlight. The cue {} had"
                    " a bad value".format(cue)
                    )

    # Label everything
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yticklabels is not None:
        plt.yticks(np.linspace(ylims[0], ylims[1], num=len(yticklabels)), 
                   yticklabels
                   )
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.show()
    
    return fig

def rate_temp(spike_dat, time_win=25, dur=None, **kwargs):
    '''
    Creates a temperature plot of neuron firing rates during a trial.

    Parameters
    ----------
    spike_dat : spikeDC
        spikeDC of one population of LIF neurons.
    time_win : float, optional
        The duration of the time window that the spikes will be 
        averaged over. The default is 25 (in ms).
    dur : float, optional
        Specifies the duration of the trial (in ms). If not specified, 
        then this will be determined by the last recorded spike time. 
        The default is None.

    Can be passed kwargs for matplotlib.pyplot.imshow.

    Returns
    -------
    fig : matplotlib.figure.Figure
       The temperature plot

    '''
    # Set default kwargs
    kwargs.setdefault('cmap', 'plasma')
    kwargs.setdefault('aspect', 'auto')
    
    neurons, times = spike_dat.get_spikes()
    
    if dur is None: dur = times[-1]
    
    #Create the temperature data
    temp_dat = np.zeros([int(round(dur - time_win)),spike_dat.size])
    for i in np.arange(int(round(dur - time_win))): 
        temp_dat[i] = spike_dat.get_spike_counts(i - time_win, i)    
    temp_dat = np.array(temp_dat, dtype='float').T
    temp_dat = np.flip(temp_dat, axis=0)
    temp_dat = temp_dat * 1000/time_win
           
    # Creates a new figure  
    fig = plt.figure()
    norm = plt.Normalize(0, temp_dat.max())
    kwargs.setdefault('norm', norm)
    
    # Plot the data
    plt.imshow(temp_dat, **kwargs)
    plt.ylabel('Neuron')
    plt.xlabel('Time (in ms)')
        
    # Adds a color bar
    cbar = fig.colorbar(cm.ScalarMappable(norm, cmap=kwargs['cmap']))
    cbar.set_label('Hertz')    
    plt.show()
        
    return fig

def pop_vector(spike_dat, time_win=50, dt=1):
    '''
    Plot the population vector over time.

    Parameters
    ----------
    spike_dat : spikeDC
        spikeDC of one population of LIF neurons.
    time_win : float, optional
        The size of the time window. The default is 50 (ms).
    dt : float, optional
        The change of time to evaluate over. The default is 1 (ms).

    Returns
    -------
    fig : matplotlib.figure.Figure
       The population vector plot.
       
    '''
    neurons, times = spike_dat.get_spikes()
    dur = int(round(times[-1]))
    arr_size = int((dur - time_win) / dt)
    
    #Create the vector and magnitude information
    mapping = np.linspace(0, 2 * np.pi, num=spike_dat.size)
    vect = np.zeros(arr_size)
    mag = np.copy(vect)
    
    for ind, time in enumerate(np.arange(time_win, dur, step=dt)): 
        vect[ind], mag[ind] = spike_dat.pop_vector(mapping, time - time_win, 
                                                   time
                                                   )
    # Corrects for the vectors being from np.pi to -np.pi
    vect = np.where(vect > 0, vect, 2 * np.pi + vect)
    mag = mag / np.max(mag)
    
    # Generate the line segments
    times = np.arange(time_win, dur, step=dt)
    points = np.vstack((times, vect)).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Creates a new figure and adds some of the labels
    fig, ax = plt.subplots()
    
    plt.xlabel('Time (in ms)')
    plt.ylabel('Vector')
    ax.set_xlim(time_win, dur)
    ax.set_ylim(np.min(mapping),np.max(mapping))
    plt.yticks(np.linspace(np.min(mapping), np.max(mapping), num = 5), 
               ('0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2', r'2$\pi$')
              )
   
    # plots the data
    norm = plt.Normalize(0, 1)
    lc = linCollect(segments, linewidths=2, cmap='binary', norm=norm)
    lc.set_array(mag)
    ax.add_collection(lc)
    cbar = fig.colorbar(lc)  
    cbar.set_label('Magnitude')

    return fig