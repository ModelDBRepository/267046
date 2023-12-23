# -*- coding: utf-8 -*-
"""
Purpose
-------
Recording and working with spiking data.

Classes
-------
spikeDC - A class for working with time spiking data sets.

Functions
---------
None

Change log
----------     
12 Feb 21 - Reorganization of code. This was split off into the
            'collect' package.

"""

from os.path import isfile
from bisect import bisect_left

import numpy as np

class spikeDC(object):
    """
    Class for working with spiking data.
    
    Attributes
    ----------
    ID : string
        A string identifier of the data.
    size : integer
        The number of neurons in the population this is
        recording from.
    raster_IDs : numpy.array(int)
        Number identifiers for the neurons.
    
    Use get_spikes() to get the neuron spikes and their
    timings.
    
    Methods
    -------
    __init__(size, ID='')
        Initializes the class to hold spiking information of
        a specific population size.
    save(file, trial, binary='binary', append=True)
        Saves a spiking data set. Has two modes of operation
        which are 'ascii' and 'binary'. By default adds to an
        existing file if it exists.
    load(file, trial, binary='binary')
        Loads a spiking data set. Has two modes of operation
        which are 'ascii' and 'binary'.
    load_trial(trial)
        Switches to the specified trial's spiking data.
    collect(spikes, time)
        Adds the spikes and their timing to the record.
    get_spikes()
        Get the raster information for population spiking.
    reset(bin_size=None)
        Resets the spiking information.
    get_spike_counts(begin=0, end=None)
        Get the spike counts during an interval.
    pop_vector(mapping, begin, end)
        Calculates the population vector over a specified period.
    covert_csv_to_npz(fname)
        Converts a spiking data from a csv format to a npz format.
        
    Use Example
    -----------
    spk_dat = spikeDC(1024,'pyr')
    
    for t in arange(0, 10):
        # CREATE SPIKES HERE
        
        spk_dat.collect(spikes, t)

    spk_dat.save('./test.npz', 1)
    
    """
   
    _default_bins = 100000
    _del = ','
    _hdr = 'trial,time,id'
    _asc_load_frmt = {'names':('trial','time','raster'),
                      'formats':('i8','f8','i8')
                      }
    _asc_save_frmt = ['%u','%.2f','%u']
        
    def __init__(self, size, ID=''):
        self._raster = None
        self._times = None
        self.reset()
        self.raster_IDs = np.array(range(1, size+1))
        self.size = size
        self.ID = ID
        self.trials = None
        
        # Typical information for file delimitation, headers, and format
        self._src_file = None
        self._spike_data = None
        
        # Internal tracking variables
        self._index = 0
        self._max_rec = self._default_bins
    
    
    def reset(self, bin_size=None):
        '''
        Empty the data collector.

        Parameters
        ----------
        bin_size : int, optional
            How large the default bin size should be. The default is 
            None, which means that it uses the default size of 100,000.
            The bin size is used to allocate space for the data while
            recording.

        Returns
        -------
        None.

        '''
        if bin_size is not None: self._default_bins = int(bin_size)
        
        self._raster = np.zeros(self._default_bins, dtype='int')
        self._times = np.zeros(self._default_bins, dtype='float')
        self._index = 0
        
    
    def load(self, file, trial, encoding='binary'):
        '''
        Load a data set.

        Parameters
        ----------
        file : string 
            Where to load the file from.
        trial : int
            The trial that should be loaded.
        encoding : string, optional
            How the data should be loaded. There are currently two
            options, which are 'ascii' and 'binary'. The default
            is 'binary'.

        Returns
        -------
        None.

        '''
        
        if (self._src_file is None) or (self._src_file != file):
            if encoding in ['b', 'bin', 'binary']:
                self._spike_data = np.load(file)
            elif encoding in ['a', 'asc', 'ascii']:            
                # Loads the data (if necessary)
                self._spike_data = np.loadtxt(file, encoding='ascii', 
                                              delimiter=self._del,
                                              dtype=self._asc_load_frmt
                                              )
            self._src_file = file
        
        # Pulls out the subset
        msk = np.where(self._spike_data['trial'] == trial)
        self._times = np.copy(self._spike_data['time'][msk])
        self._raster = np.copy(self._spike_data['raster'][msk])
        self._index = len(self._times)
        self.trials = np.unique(self._spike_data['trial'])
        
    def load_trial(self, trial):
        '''
        Load a trial from the previously loaded data file.

        Parameters
        ----------
        trial : int
            The trial that should be loaded.

        Returns
        -------
        None.

        '''
        
        if (self._src_file is None):
            raise FileNotFoundError('No file has been loaded by this class',
                                    'yet. Use the load function first.')
        
        # Pulls out the subset
        msk = np.where(self._spike_data['trial'] == trial)
        self._times = np.copy(self._spike_data['time'][msk])
        self._raster = np.copy(self._spike_data['raster'][msk])
        self._index = len(self._times)
        
    def _resize_rec(self, size):
        '''
        Resizes the record.

        Parameters
        ----------
        size : int
            The size of the new record.

        Returns
        -------
        None.

        '''
        to_add = size - self._max_rec
        self._raster = np.hstack((self._raster, 
                                 np.zeros(to_add, dtype='int'))
                                )
        self._times = np.hstack((self._times, 
                                 np.zeros(to_add, dtype='float'))
                                )
        self._max_rec = size
    
    
    def collect(self, spikes, time):
        '''
        Adds the spikes and their timing to the record.

        Parameters
        ----------
        spikes : numpy.array(bool)
            List of the neurons and whether they spiked.
        time : float
            The time of the spike.

        Returns
        -------
        None.

        '''
        
        # Creates the data for the raster
        spk = spikes * self.raster_IDs
        spk = spk[np.where(spk != 0)]
        
        end_ind = self._index + spk.shape[0]
        
        # If necessary, adjusts the size of the record
        if end_ind >= self._max_rec: 
            self._resize_rec(self._max_rec + self._default_bins)
        
        # Replaces the zeros with values
        self._raster[self._index:end_ind] = spk
        self._times[self._index:end_ind] = time
        self._index = end_ind
    
    
    def pop_vector(self, mapping, begin, end, return_counts=False):
        '''
        Calculates the population vector over a period of time.

        Parameters
        ----------
        mapping : np.array(float)
            Relationship between neuron IDs and direction.
        begin : float
            When to start the vector calculation.
        end : float
            When to end the vector calculation

        Returns
        -------
        vector : float
            The direction of the population firing.
        magnitude : float
            How strongly concentrated the population firing is.

        '''
        # pull out the important information and get their direction
        counts = self.get_spike_counts(begin, end)
        x_change = np.sum(np.cos(mapping)*counts)
        y_change = np.sum(np.sin(mapping)*counts)
        vector = float(np.arctan2(y_change,x_change))
        magnitude = float((x_change**2 + y_change**2)**0.5)
        
        if return_counts:
            return vector, magnitude, counts        
        else:
            return vector, magnitude
    
    
    def get_spike_counts(self, begin=0, end=None):
        '''
        Get the spike counts during an interval [begin, end).

        Parameters
        ----------
        begin : float, optional
            Beginning of the interval (inclusive). The default is 0.
        end : float, optional
            End of the interval (non-inclusive). The default is None.

        Returns
        -------
        counts : numpy.array(int)
            A list of the spike counts of each neuron during
            the interval.

        '''
        if end is None: end = self._times[-1]
        
        # Extract the relevant data information
        raster, times = self.get_spikes()
        subset = raster[bisect_left(times, begin):
                        bisect_left(times, end)]
        
        # organize the spikes into the spikes by ID
        mat = np.transpose(np.ones([subset.shape[0], 
                                    self.raster_IDs.shape[0]
                                    ]) 
                           * self.raster_IDs
                           )
        mat = mat - subset
        mat = np.where(mat == 0, 1, 0)
        counts = np.sum(mat, axis = 1)
               
        return counts
    
    def get_max_time(self):
        '''
        Gets the largest time value in the data across all trials.

        Returns
        -------
        max_time
            The largest time for all trials.

        '''
        
        return np.max(self._spike_data['time'])
        
    def save(self, fname, trial=0, encoding='binary', append=True):
        '''
        Saves the current trial data.

        Parameters
        ----------
        fname : string
            Where to save the data.
        trial : int, optional
            Trial number that will be output. The default is 0.
        encoding : string, optional
            How the data an be saved. There are currently two
            options, which are 'ascii' and 'binary'. The 
            default is 'binary'.
        append : bool, optional
            How the data should be saved. There are currently two
            options, which are 'ascii' and 'binary'. The default
            is 'binary'.

        Returns
        -------
        None.

        '''
        neurs, times = self.get_spikes()
        
        # creates the data to be worked with and arranges it vertically
        dat = np.vstack((np.ones([times.shape[0],]) * trial,
                                 times,
                                 neurs)
                        ).T
        
        # Appends the new data to existing data
        if isfile(fname) and append:
            if encoding in ['b', 'bin', 'binary']:
                old = np.load(fname)
            elif encoding in ['a', 'asc', 'ascii']:            
                # loads the old data, adds the new data, and then writes
                old = np.loadtxt(fname, encoding='ascii',
                                 delimiter=self._del,
                                 dtype=self._asc_load_frmt
                                 )
            old = np.vstack((old['trial'],
                             old['time'],
                             old['raster'])
                            ).T
            dat = np.vstack((old, dat))
        
        
        # Saves the data
        if encoding in ['b', 'bin', 'binary']:
            np.savez_compressed(fname, 
                                trial=dat[:,0], 
                                time=dat[:,1], 
                                raster=dat[:,2]
                                )
        elif encoding in ['a', 'asc', 'ascii']: 
            np.savetxt(fname, dat, delimiter=self._del,
                       fmt=self._asc_save_frmt,
                       header=self._hdr,
                       encoding='ascii'
                       )
            
    def get_spikes(self):
        '''
        Get the raster information for population spiking.

        Returns
        -------
        neurons : numpy.array('int')
            Which neurons fired at the specific times.
        times : numpy.array('float')
            When neurons fired.

        '''
        neurons = np.copy(self._raster)
        times = np.copy(self._times)
        
        neurons = neurons[:self._index]
        times = times[:self._index]
        
        return neurons, times
        
    def convert_csv_to_npz(self, fname):
        '''
        Converts a csv file into an npz file.

        Parameters
        ----------
        fname : string
            Name and location of the file that needs converted.

        '''
        print('Loading ascii file', fname)
        self.load(fname, 1, encoding='ascii')

        new_name = fname[:-4]
        
        output = np.vstack((self._spike_data['trial'],
                 self._spike_data['time'],
                 self._spike_data['raster'])
                 ).T
        
        print('Saving binary version of', fname, 'as', new_name + '.npz')
        np.savez_compressed(new_name, trial=output[:,0], 
                                      time=output[:,1], 
                                      raster=output[:,2]
                            )   

'''
    
# The following class is less efficient than SpikeDC, but is retained for
#   backward compatibility with code during 2019. Most of the files created
#   through this class have been converted to .npz files using the
#   spikeDC.convert_asc_to_bin function.
class SpikeDataCollector(object):
    _del = ','
    _hdr = 'trial,time,id'
    _frmt = ['%u','%.2f','%u']
        
    def __init__(self, size, ID = ''):
        self.raster, self.rasterTimes = [], []
        self.rasterIDs = np.array(range(1, size+1))
        self.size = size
        self.ID = ID
        
        # Typical information for file delimitation, headers, and format
        self._file = None
        self._spikeData = None
    
    
    def resetSpikes(self):
        self.raster, self.rasterTimes = [], []
        
    
    def loadSpikeInfo(self, fname, trialNum):
        self.resetSpikes()
    
        # Loads the data (if necessary)
        if (self._file is None) or (self._file != fname):
            self._spikeData = np.loadtxt(fname, delimiter=self._del)
            self._file = fname
        
        # Pulls out the subset
        trialDat = np.copy(self._spikeData[self._spikeData[:,0] == trialNum])
        self.rasterTimes = np.copy(trialDat[:,1])
        self.raster = np.copy(trialDat[:,2])
        
    
    def collectSpikes(self, spikes, currTime):
        # Creates the data for the raster plot
        spk = spikes * self.rasterIDs
        spk = spk[spk != 0]
        self.raster = np.hstack((self.raster, spk))
        self.rasterTimes = np.hstack((self.rasterTimes, 
                                      (np.ones([spk.shape[0]],) * currTime))
                                     )
    
    
    def getPopVector(self, mapping, beginTime, endTime):
        # pull out the important information and get their direction
        counts = self.getSpikeCounts(beginTime, endTime)
        nRates = counts / np.sum(counts)
        xChange = np.sum(np.cos(mapping)*nRates)
        yChange = np.sum(np.sin(mapping)*nRates)
                
        return (float(np.arctan2(yChange,xChange)), 
                float((xChange*xChange + yChange*yChange)**0.5)
                )    
    
    
    def getSpikeCounts(self, beginTime = 0, endTime = None):
        if endTime is None: endTime = self.rasterTimes[-1]
        
        # pull out the important information
        subset = np.where((self.rasterTimes <= endTime) 
                          & (self.rasterTimes >= beginTime),self.raster, 0)
        subset = np.array(subset[subset != 0], dtype = 'int')
        
        # organize the spikes into the spikes by the neuron ID
        valueMatrix = np.transpose(
                        np.ones([subset.shape[0], self.rasterIDs.shape[0]]) 
                        * self.rasterIDs)
        valueMatrix = valueMatrix - subset
        valueMatrix = np.where(valueMatrix == 0, 1, 0)
        counts = np.sum(valueMatrix, axis = 1)
               
        return counts
    
    def outputTrial(self, fname, trialNum = 0):
        # creates the data to be worked with and arranges it vertically
        dat = np.vstack((np.ones([self.rasterTimes.shape[0],]) * trialNum, 
                         self.rasterTimes, 
                         self.raster)
                        ).T
        
        # checks to see if the file exists (needs to know if it should append)
        if isfile(fname):
            # loads the old data, adds the new data, and then writes
            old = np.loadtxt(fname, delimiter=self._del)
            dat = np.vstack((old, dat))
            np.savetxt(fname, dat, delimiter=self._del, 
                       fmt=self._frmt, header=self._hdr)            
        else:
            # just saves the file if one doesn't exist yet
            np.savetxt(fname, dat, delimiter=self._del, 
                       fmt=self._frmt, header=self._hdr)
'''
            