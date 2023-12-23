# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:11:30 2019
@author: ocalvin

18 Sept 2019 - Version 2 of the network code started. The goal is to move away 
               from separate neuron and network classes and to instead 
               integrate them into a singular entity. The primary reason for 
               this update is that the NumPy code is better optimized, so it 
               will be better to move to a matrix algebraic approach that will 
               handle the neurons as a population.

20 Sept 2019 - Optimized :)

21 Oct 2019 - Had to rework how spikes are collected and processed because 
              the previous method was incompatible with multiple networks. 
              It's a bit slower now, but I may be able to improve on this 
              later.
              
18 Sep 2020 - Went back and updated the documentation so that this is better
              for future use.

22 Apr 2021 - Added more documentation and established hierarchical code
              organization.
"""

import numpy as np

from . import currents
from utils import ivprobs as rk

class baseLIFPop(object):
    def __init__(self, size, vTh=-50, vReset=-60, refDur=2, cM=0.5,
                 AMPA_g=0.7, NMDA_g=0.07, GABA_g=0.1, 
                 noise_rate=1.8, noise_g=0.003, leak_g=0.025, leak_vL=-70,
                 record_i=False):
        '''
        Initialize the leaky-integrate-and-fire population.

        Parameters
        ----------
        size : int
            The size of the population.
        vTh : float, optional
            The membrane voltage threshold that will produce a spike. 
            The default is -50 (mV).
        vReset : float, optional
            The resting membrane voltage. The default is -60 (mV).
        refDur : float, optional
            The duration of the refractory period. The default is 2 (ms).
        cM : float, optional
            The membrane capacitance. The default is 0.5 (nF).
        AMPA_g : float, optional
            AMPA receptor conductance. The default is 0.7 (µS).
        NMDA_g : float, optional
            NDMA receptor conductance. The default is 0.07 (µS).
        GABA_g : float, optional
            GABA receptor conductance. The default is 0.1 (µS).
        noise_rate : float, optional
            Rate of Poisson spikes. The default is 1.8 (kHz).
        noise_g : float, optional
            AMPA noise receptor conductance. The default is 0.003 (µS).
        leak_g : float, optional
            The conductance of membrane leakage. The default is 0.025 (µS).
        leak_vL : float, optional
            Leak reversal potential. The default is -70.
        record_i : bool, optional
            Whether the current should be recorded. The default is False.

        Returns
        -------
        None.

        '''
        
        # Population parameters
        self.size = int(size)       # How many neurons are in the population       
        self.vTh = float(vTh)       # Spike-initiation threshold (in mV)
        self.vReset = float(vReset) # Resting potential (in mV)
        self.refDur = float(refDur) # Duration of the refractory period (in ms)       
        self.cM = float(cM)         # Membrane capacitance (in nF)

        # Blank lists
        self._neuronBaseList = np.array(range(self.size), dtype = 'int')
        self._blankSpikes = np.zeros([self.size,], dtype='bool')
        self._blankTimes = np.zeros([self.size,], dtype='float')
        
        # Current Sources
        self.AMPA = currents.AMPAR(self.size, 
                                   record_i=record_i, 
                                   g=AMPA_g)
        self.NMDA = currents.NMDAR(self.size, 
                                   record_i=record_i, 
                                   g=NMDA_g)
        self.GABA = currents.GABAR(self.size, 
                                   record_i=record_i, 
                                   g=GABA_g)
        self.noise = currents.noisePoisson(self.size,
                                           g=noise_g, 
                                           rate=noise_rate)
        self.leak = currents.leak(self.size, 
                                  record_i=record_i, 
                                  g=leak_g, 
                                  vL=leak_vL)
        self.afferents = currents.affFlat(self.size)
            
        # Initializes the membrane potential
        self._vM = (np.random.rand(self.size) 
                    * (self.vTh - self.vReset)
                    + self.vReset)
        self._tempvM = np.zeros([self.size,], dtype = 'float')
        self._spiked = np.zeros([self.size,], dtype='bool')
        self._spikeTime = np.zeros([self.size,], dtype='float')
        self._endRefract = np.zeros([self.size,], dtype='float')
        
        # Arrays for temporary information
        self.numSpikes = 0
              
        # Received spikes
        self._rPyrCount, self._rIntCount = 0, 0
        self._rEmpty = np.empty((0, self.size+1))
        self._rPyrSpikes = np.copy(self._rEmpty)
        self._rIntSpikes = np.copy(self._rEmpty) 
    
    # ------------- State Reset Command -------------
    def reset(self):
        '''
        Resets the population to a blank slate by reinitializing
        many internal variables and classes.
        
        '''
        # Resets the membrane
        self._vM = (np.random.rand(self.size) 
                    * (self.vTh - self.vReset)
                    + self.vReset)
        self._tempvM = np.zeros([self.size,], dtype = 'float')
        self._spiked = np.zeros([self.size,], dtype='bool')
        self._spikeTime = np.zeros([self.size,], dtype='float')
        self._endRefract = np.zeros([self.size,], dtype='float')
        
        # Resets the currents
        self.AMPA.reset()
        self.NMDA.reset()
        self.GABA.reset()
        self.noise.reset()
        self.leak.reset()
        self.afferents.reset()

    #--------------- Network Input Functions ------------
    
    def stim(self, **kwargs):
        '''
        Add stimulation to the network. Can set the afferent
        current via relevant keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the 'afferents'
            parameter's stim function.

        '''
        self.afferents.stim(**kwargs)
        
    def removeStim(self): 
        '''
        Remove afferent current.
        
        '''
        self.afferents.hide()
        
    def receivePyrSpikes(self, dt, num, dts, weights):
        '''
        Receive excitatory spikes from pyramidal cells.

        Parameters
        ----------
        dt : float
            The change in time.
        num : int
            The number of spikes.
        dts : np.array(float)
            List of spike times.
        weights : np.array(float)
            Weights for each of the spikes.

        '''
        if num > 0:
            # arranges the received spikes and cuts those that don't matter
            temp = np.hstack((dts, weights))
            temp = np.delete(temp, np.argwhere(temp[:,0] == dt), axis = 0)
            
            # adds these spikes to the other ones that have occured
            self._rPyrSpikes = np.vstack((self._rPyrSpikes, temp))
            
        self._rPyrCount = self._rPyrSpikes.shape[0]

    def receiveIntSpikes(self, dt, num, dts, weights):
        '''
        Receive inhibitory spikes from interneurons.

        Parameters
        ----------
        dt : float
            The change in time.
        num : int
            The number of spikes.
        dts : np.array(float)
            List of spike times.
        weights : np.array(float)
            Weights for each of the spikes.

        '''
        if num > 0:
            # organizes the received spikes 
            temp = np.hstack((dts, weights))
            temp = np.delete(temp, np.argwhere(temp[:,0] == dt), axis = 0)
            
            # adds these spikes to the other ones that have occured this time step
            self._rIntSpikes = np.vstack((self._rIntSpikes, temp))
            
        self._rIntCount = self._rIntSpikes.shape[0]
            
    #--------------- Processing Functions ------------   
    
    def predictSpikes(self, dt, time):
        '''
        Uses the second-order Runge-Kutta to determine if a neuron
        is likely to spike within the duration.

        Parameters
        ----------
        dt : float
            The change in time.
        time : float
            The current time.

        '''
        # Resets the received pyramidal and interneuron spikes
        self._rPyrSpikes = np.copy(self._rEmpty)
        self._rIntSpikes = np.copy(self._rEmpty)
        self._rPyrCount, self._rIntCount = 0, 0
        
        # Calculates the anticipated membrane voltage
        # Note: This isn't perfectly accurate because the refractory period 
        #    could be between time and endTime, but this is more 
        #    computationally efficient and the possible error is minor.
        self._tempvM = (self._vM + 
                        (rk.RK2(self.project_dVmdt, 0, self._vM, dt)
                         /self.cM)
                        )
        
        # Handles if a neuron spiked in that interval
        self._spiked = np.where(self._tempvM >= self.vTh, True, False)
        self._spikeTime = np.where(self._spiked == True, 
                                      (dt 
                                       * (self.vTh - self._vM)  
                                       / (self._tempvM - self._vM)
                                      ), 
                                   dt
                                   )
        self._endRefract = np.where(self._spiked == True, 
                                    time + self._spikeTime + self.refDur, 
                                    self._endRefract
                                    )
        self.numSpikes = np.sum(self._spiked)
        
        # Makes spikeTimes verticle so that receiving populations don't need to
        self._spikeTime = np.reshape(self._spikeTime, (-1,1))
        
    def processTimeStep(self, dt, time):
        '''
        Processes the spikes that have been recently received.

        Parameters
        ----------
        dt : float
            The change in time.
        time : float
            The current time.

        Returns
        -------
        None.

        '''
        pyrTimes, pyrWeights = np.array([]), np.array([])
        intTimes, intWeights = np.array([]), np.array([])
        
        # Must resort the received spike times, because this could be 
        #     misstacked if there are more than one source
        if self._rPyrCount > 0: 
            self._rPyrSpikes = self._rPyrSpikes[np.argsort(self._rPyrSpikes[:,0])]
            pyrTimes = self._rPyrSpikes[:,0]
            pyrWeights = self._rPyrSpikes[:, 1:]
        if self._rIntCount > 0: 
            self._rIntSpikes = self._rIntSpikes[np.argsort(self._rIntSpikes[:,0])]
            intTimes = self._rIntSpikes[:,0]
            intWeights = self._rIntSpikes[:, 1:]

        # update the currents
        self.AMPA.update(dt, pyrTimes, pyrWeights)
        self.NMDA.update(dt, pyrTimes, pyrWeights)
        self.GABA.update(dt, intTimes, intWeights)
        self.noise.update(dt, 0, 0)
        self.leak.update(0, 0, 0)
        self.afferents.update(dt, 0, 0)
    
        # Calculates the new voltage of each cell 
        self._vM = np.where(self._endRefract < time, 
                            self._vM 
                            + (rk.RK2(self.dVmdt, 0, self._vM, dt) / self.cM),
                            self.vReset
                            )
        
    def dVmdt(self, dt, vM):
        '''
        Change of the membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        Raises
        ------
        NotImplementedError
            This component of the base class is not defined.

        '''
        raise NotImplementedError("You are trying to invoke 'dVmdt' with the base class.")
    
    def project_dVmdt(self, dt, vM):
        '''
        Given current parameters, what is the expected change of the 
        membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        Raises
        ------
        NotImplementedError
            This component of the base class is not defined.
            
        '''
        raise NotImplementedError("You are trying to invoke 'project' with the base class.")
        
    # ------------- API Commands ---------------
    
    def VMs(self): 
        '''
        Get the membrane potential voltages of all neurons.

        Returns
        -------
        np.array(float)
            The membrane potential voltages.

        '''
        return np.copy(self._vM)
        
    def spikes(self): 
        '''
        Get a boolean list of whether a set of neurons spiked.

        Returns
        -------
        np.array(bool)
            List of whether a set of neurons spiked.

        '''
        return np.copy(self._spiked)
    
        # Returns a copy of the spikes, which will be great for rastergrams
    def spikeTimes(self): 
        '''
        Get a list of the times during that neuron's spiked during 
        the last dt.

        Returns
        -------
        np.array(float)
            List of when during the last dt the neuron's spiked.

        '''        
        return np.copy(self._spikeTime)

# ----------------------------------------------------------------------      
# ----------------------- Typical Populations --------------------------
# ---------------------------------------------------------------------- 

class pyrPop(baseLIFPop):
    def __init__(self, size, **kwargs):
        '''
        Initialize a pyramidal cell LIF population.

        Parameters
        ----------
        size : int
            The size of the population.
        See baseLIFPop for all keyword arguments.

        '''
        kwargs.setdefault('vTh', -52)
        kwargs.setdefault('vReset', -60)
        
         # creates the population           
        super().__init__(size, **kwargs)

    # Rate of change for the membrane potential's voltage
    def project_dVmdt(self, dt, vM):
        '''
        Given current parameters, what is the expected change of the 
        membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.
            
        '''
        return (-self.afferents.project(dt, vM) 
                - self.noise.project(dt, vM) 
                - self.leak.project(dt, vM) 
                - self.AMPA.project(dt, vM) 
                - self.NMDA.project(dt, vM) 
                - self.GABA.project(dt, vM)
                )
       
    # Rate of change for the membrane potential's voltage
    def dVmdt(self, dt, vM):
        '''
        Change of the membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        '''
        return (-self.afferents.current(dt, vM) 
                - self.noise.current(dt, vM) 
                - self.leak.current(dt, vM) 
                - self.AMPA.current(dt, vM) 
                - self.NMDA.current(dt, vM) 
                - self.GABA.current(dt, vM)
                )

class interPop(baseLIFPop):
    def __init__(self, size, **kwargs):
        '''
        Initialize an interneuron LIF population.

        Parameters
        ----------
        size : int
            The size of the population.
        See baseLIFPop for all keyword arguments.

        '''
        # sets the default parameters if they are not included in kwargs
        kwargs.setdefault('cM', 0.2)
        kwargs.setdefault('vTh', -52)
        kwargs.setdefault('vReset', -60)
        kwargs.setdefault('refDur', 1)        

        kwargs.setdefault('leak_g', 0.02)
        kwargs.setdefault('leak_vL', -65)
        
        kwargs.setdefault('AMPA_g', 0.2)
        kwargs.setdefault('NMDA_g', 0.02)
        
        # creates the population       
        super().__init__(size, **kwargs)
        
    # Rate of change for the membrane potential's voltage
    def project_dVmdt(self, dt, vM):
        '''
        Given current parameters, what is the expected change of the 
        membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.
            
        '''
        return (-self.afferents.project(dt, vM) 
                - self.noise.project(dt, vM) 
                - self.leak.project(dt, vM) 
                - self.AMPA.project(dt, vM) 
                - self.NMDA.project(dt, vM)
                )
       
    # Rate of change for the membrane potential's voltage
    def dVmdt(self, dt, vM):
        '''
        Change of the membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        '''
        return (-self.afferents.current(dt, vM) 
                - self.noise.current(dt, vM) 
                - self.leak.current(dt, vM) 
                - self.AMPA.current(dt, vM) 
                - self.NMDA.current(dt, vM)
                )

class pyrRingPop(baseLIFPop):
    def __init__(self, size, **kwargs):
        '''
        Initialize a circular pyramidal cell LIF population.

        Parameters
        ----------
        size : int
            The size of the population.
        See baseLIFPop for all keyword arguments.

        '''
        # sets the default parameters if they are not included in kwargs
        kwargs.setdefault('leak_g', 0.025)
        kwargs.setdefault('AMPA_g', 0)
        kwargs.setdefault('NMDA_g', 0.4)
        kwargs.setdefault('GABA_g', 1.40)
        kwargs.setdefault('noise_g', 0.0031)
        super().__init__(size, **kwargs)
        
        # Creates the positional information based on the size of the ring.
        self.directions = np.linspace(0, np.pi * 2, num=size, endpoint=False)
        
        # Makes afferent signals directional                   
        self.afferents = currents.affRing(self.size, self.directions)

    def project_dVmdt(self, dt, vM):
        '''
        Given current parameters, what is the expected change of the 
        membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        '''
        return (-self.afferents.project(dt, vM) 
                - self.noise.project(dt, vM) 
                - self.leak.project(dt, vM) 
                - self.AMPA.project(dt, vM) 
                - self.NMDA.project(dt, vM) 
                - self.GABA.project(dt, vM)
                )
       
    def dVmdt(self, dt, vM):
        '''
        Change of the membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.
            
        '''
        return (-self.afferents.current(dt, vM) 
                - self.noise.current(dt, vM) 
                - self.leak.current(dt, vM) 
                - self.AMPA.current(dt, vM) 
                - self.NMDA.current(dt, vM) 
                - self.GABA.current(dt, vM)
                )
    
    def stim(self, direction, **kwargs):
        '''
        Add stimulation to the network. Can set the afferent
        current via relevant keyword arguments.

        Parameters
        ----------
        direction : float
            The direction that the stimulus is associated with.
        **kwargs : dict
            Keyword arguments to be passed to the 'afferents'
            parameter's stim function.

        '''
        self.afferents.stim(direction, **kwargs)
    

class interRingPop(pyrRingPop):
    def __init__(self, size, **kwargs):
        '''
        Initialize a circular interneuron LIF population.

        Parameters
        ----------
        size : int
            The size of the population.
        See baseLIFPop for all keyword arguments.

        '''
        # sets the default parameters if they are not included in kwargs
        # positional information and directional afferents are created in pyrRingPop
        kwargs.setdefault('cM', 0.2)
        kwargs.setdefault('leak_g', 0.02)
        kwargs.setdefault('AMPA_g', 0)
        kwargs.setdefault('NMDA_g', 0.3)
        kwargs.setdefault('GABA_g', 1.05)
        kwargs.setdefault('noise_g', 0.00238)
        kwargs.setdefault('refDur', 1)
        super().__init__(size, **kwargs)                           
        
    # Rate of change for the membrane potential's voltage
    def project_dVmdt(self, dt, vM):
        '''
        Given current parameters, what is the expected change of the 
        membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.
            
        '''
        return (-self.afferents.project(dt, vM) 
                - self.noise.project(dt, vM) 
                - self.leak.project(dt, vM) 
                - self.AMPA.project(dt, vM) 
                - self.NMDA.project(dt, vM) 
                - self.GABA.project(dt, vM)
                )
       
    # Rate of change for the membrane potential's voltage
    def dVmdt(self, dt, vM):
        '''
        Change of the membrane potential's voltage over time.

        Parameters
        ----------
        dt : float
            The change in time.
        vM : float
            The membrane potential's voltage.

        '''
        return (-self.afferents.current(dt, vM) 
                - self.noise.current(dt, vM) 
                - self.leak.current(dt, vM) 
                - self.AMPA.current(dt, vM) 
                - self.NMDA.current(dt, vM) 
                - self.GABA.current(dt, vM)
                )    
        
