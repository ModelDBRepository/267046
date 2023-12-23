# -*- coding: utf-8 -*-
"""
Purpose
-------
This code implements receptor currents for realistic spiking neuron models.

Classes
-------
currentBase - Base class for current sources within a neural population.
   leak - Leaky currents
   noiseGaussian - Provides random Gaussian noise
   noisePoisson - Provides random Poisson noise
   AMPAR - Glutamatergic AMPA receptors.
   NMDAR - Glutamatergic NMDA receptors.
   GABAR - GABA receptors.
   affFlat - Afferent current that equally affects the entire population
             and is steady over time.
   affRing - Afferent current that unequally affects the entire population
             and is steady over time. 
   affFlatPoisson - Afferent current that affects the entire population
             and is unsteady over time. Uses a Poisson process to generate
             the afferent current.
   affRingPoisson - Afferent current that unequally affects the entire 
             population and is unsteady over time. Uses a Poisson process 
             to generate the afferent current.

Functions
---------
None

Notes
-----
The current classes are used via few different functions that subserve
different purposes. These functions are 'project', 'update', and 'current',
which should be used in that order. Project uses the current trajectory of
the population's current to guess at what the value will be in the future.
Update takes information about spiking times and applies them to the
population. Current is used to reference the currents after they have been
updated.

Change log
----------     
03 May 20 - Cleaned up the code to be more Pythonic and added more
            documentation. Final version for eventual upload.
            
18 Sep 20 - Split out the kinetics from the currents.

26 Feb 21 - Reviewed documentation and made updates.
            
References
----------
[1] Wang, X.J. (1999). Synaptic Basis of Cortical Persistent 
    Activity:  the Importance of NMDA Receptors to Working Memory. 
    The Journal of Neuroscience, 19(21), 9587–9603. 
    doi: 10.1523/JNEUROSCI.19-21-09587.1999

[2] Compte, A., Brunel, N., Goldman-Rakic, P.S., & Wang, X.J. (2000). 
    Synaptic mechanisms and network dynamics underlying spatial 
    working memory in a cortical network model. Cerebral Cortex, 10,
    910-923. doi: 10.1093/cercor/10.9.910 
     
"""

import numpy as np

from .kinetics import firstOrdKinExp, secondOrdKin

class currentBase(object):
    """
    Base class for current sources within a neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    record_i : boolean
        Whether the mean current should be recorded.
    record_rate : float
        The sampling rate (in Hz) that the current should be recorded. 
    curr_record : list(float)
        If record_i is True, then this is the recorded mean current. 
    
    Methods
    -------
    __init__(size, record_i=False, record_rate=2000)
        Initializes currentBase.
    project(dt, vM)
        Predicts future current values.
    update(dt, spk_times, weights)
        Updates the currents.
    current(dt, vM)
        Gets the currents' value.
    record_i(dt)
        Records the current.
    reset()
        Resets the currents and tracking.
    
    """
    
    def __init__(self, size, record_i=False, record_rate=2000):
        """
        Initializes the currentBase class.

        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        record_i: boolean, optional
            Whether the mean current should be recorded. The default is False.
        record_rate: float, optional
            The sampling rate (in Hz) that the current should be recorded. 
            The default is 2000 Hz (i.e., a 0.5 ms sampling interval).

        """
        
        self.size = size
        
        # Currents at t and t-dt
        self._curr = np.zeros([self.size,], dtype = 'float')
        self._prev_curr = np.copy(self._curr)
        
        #tracking variables
        self.record = record_i
        self.record_rate = record_rate #in Hz
        self._interval = 1/self.record_rate
        self._block_t = 0
        self._curr_block = []
        self.curr_record = []
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The current membrane voltage potential.

        Raises
        ------
        NotImplementedError
            This function is not implemented in the currentBase.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """        
        raise NotImplementedError(
            'Trying to call "project" with the currentBase class')
        
    def update(self, dt, spk_times, weights):
        """
        Updates the current based on spiking and the connection 
        weights.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        Raises
        ------
        NotImplementedError
            This function is not implemented in the currentBase.
            
        """        
        raise NotImplementedError(
            'Trying to call update with the currentBase class')
        
    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """        
        if dt == 0: return np.copy(self._prev_curr)
        return np.copy(self._curr) 
    
    def record_i(self, dt):
        '''
        Records the current change.

        Parameters
        ----------
        dt : float
            The change in time.

        '''        
        if self.record_i:
            self._curr_block.append(np.mean(self._curr))
            self._block_t += dt
            
            if self._block_t >= self._interval:
                self.curr_record.append(np.mean(self._curr_block))
                self._curr_block = []
                self._block_t = 0    
    
    def reset(self): 
        """
        Resets the current and tracking for a new run.
        
        """        
        self._curr = np.zeros([self.size,], dtype = 'float')
        self._prev_curr = np.copy(self._curr)
        self._block_t=0
        self._curr_block = []
        self.curr_record = []

        
# -------------------- Membrane Currents -----------------------------

class leak(currentBase):
    """
    Leaky currents for a neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    g : float
        Conductance of the leaking membrane..
    vL : float
        The resting potential of the neurons.
    rand_sd : float
        Standard deviation of conductances if they are randomized. 
    
    Methods
    -------
    __init__(size)
        Initializes leak.
    current(dt, vM)
        Gets the currents' value.
    project(dt, vM)
        Predicts future current values.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
        
    """
    
    def __init__(self, size, g=0.025, vL=-70, rand_sd=None, **kwargs):
        """
        Initializes the leak class.

        Parameters
        ----------
        size : uint
            The size of the neural population.
        g : float, optional
            Conductance of the leaking membrane. The default is 0.025 uS.
        vL : float, optional
            The resting potential of the neurons. The default is -70 mV.
        rand_sd : float, optional
            The standard deviation of the conductances. The default is
            None. This parameter exists to be able to replicate some
            X.J. Wang (1999) simulations.

        """
        super().__init__(size, **kwargs)
        
        # Default current parameters
        self.g = g                        # Conductance (in uS)
        self.vL = vL                      # Resting potential (in mV)
        if rand_sd is not None:
            self._randomG = True
            self.rand_sd = rand_sd
            self.g += np.random.randn(self.size) * self._randSD
        else:
            self._randomG = False
            self.rand_sd = 0
        
    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        if self._randomG: 
            self.g += np.random.randn(self.size) * self._randSD
    
    def project(self, dt, vM): 
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """        
        return (self.g * (vM - self.vL))
        
    def update(self, dt, spikes, weights):
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        """        
        self._prev_curr = np.copy(self._curr)

    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """        
        self._curr = (self.g * (vM - self.vL))
        if dt != 0: 
            self.record_i(dt)
        return np.copy(self._curr)
# -------------------- Noise Currents -----------------------------

class noiseGaussian(currentBase):
    """
    A current source that provides random Gaussian noise to a 
    neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    aff_m : float
        Mean of the afferent noise.
    aff_sd : float
        Standard deviation of the afferent noise.
    
    Methods
    -------
    __init__(size, mean, sd, **kwargs)
        Initializes noiseGaussian class.
    project(dt, vM)
        Predicts future current values.
    update(dt, spk_times, weights)
        Updates the currents.
    current(dt, vM)
        Gets the currents' value.
    reset()
        Resets the currents and tracking.
    
    """
    
    def __init__(self, size, mean, sd, **kwargs):
        """
        Initializes the noiseGaussian class.

        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        mean : float
            The mean of the afferent current.
        sd : float
            The standard deviation of the afferent current.
        
        """
        super().__init__(size, **kwargs)
        
        self.aff_m = mean
        self.aff_sd = sd
        
        # Whether a new set of random numbers should be drawn
        self._draw = True   
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """
        
        # Creates a new random number if it needs to
        if self._draw:
            self._prev_curr = np.copy(self._curr)
            self._curr = self.aff_m + np.random.randn(self.size) * self.aff_sd
            self._draw = False
        
        if dt == 0: 
            return np.copy(self._prev_curr)
        else:
            return np.copy(self._curr)
        
    def update(self, dt, spikes, weights): 
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        """
        self._draw = True
        self.record_i(dt)

class noisePoisson(currentBase):
    """
    A current source that provides random Poisson noise to a 
    neural population.
    
    Attributes
    ----------   
    size : uint
        Size of the neuronal population.
    g : float, optional
        The conductance of the AMPA-mediated current.
    vE : float, optional 
        The resting potential of these receptors.
    alpha : float, optional
        The increase in the kinetic per received spike.
    tau : float, optional
        The kinetic's average life expectancy.
    rate : float, optional
        The rate of external firing.
        
    Methods
    -------
    __init__(size, mean, sd, **kwargs)
        Initializes noiseGaussian class.
    project(dt, vM)
        Predicts future current values.
    update(dt, spk_times, weights)
        Updates the currents.
    current(dt, vM)
        Gets the currents' value.
    reset()
        Resets the currents and tracking.
    
    """
    def __init__(self, size, g=0.003, vE=0, 
                 alpha=1, tau=2, rate=1.8, **kwargs):
        """
        Initializes the noisePoisson class that was proposed in [2]. 
        For alpha, this uses the AMPA equation mentioned in [2].
        
        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        g : float, optional
            The conductance of the AMPA-mediated current. The default
            is 0.003.
        vE : float, optional 
            The resting potential of these receptors. The default is 0.
        alpha : float, optional
            The increase in the kinetic per received spike. The default
            is 1.
        tau : float, optional
            The kinetic's average life expectancy. The default is 2 ms.
        rate : float, optional
            The rate of external firing.  The default is 1.8 kHz.

        [2] Compte, A., Brunel, N., Goldman-Rakic, P.S., & Wang, X.J. (2000). 
            Synaptic mechanisms and network dynamics underlying spatial 
            working memory in a cortical network model. Cerebral Cortex, 10,
            910-923. doi: 10.1093/cercor/10.9.910
            
        """        
        super().__init__(size, **kwargs)
       
        # Set parameters
        self.g = g
        self.vE = vE
        self.alpha = alpha
        self.tau = tau
        self.rate = rate
        
        # Whether a new set of random numbers should be drawn
        self._draw = True                                    
        
        # Kinetic tracking
        self._s = np.zeros([self.size,], dtype = 'float')    # AMPA kinetic
      
    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self._s = np.zeros([self.size,], dtype = 'float')
    
    def _decay(self, dt):
        """
        Decays the AMPA kinetic.

        Parameters
        ----------
        dt : float
            The amount of time that has elapsed since the last
            update.

        """
        # proportion to decay
        p = np.exp(-dt/self.tau)
        self._s *= p
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """
        
        # Creates a new random number if it needs to
        if (self._draw) & (dt != 0):
            self._prev_curr = np.copy(self._curr)
            nseTimes = np.random.rand(self.size) * dt
            
            self._decay(nseTimes)
            self._s += (self.alpha 
                        * np.random.poisson(dt * self.rate, self.size)
                        )
            self._decay(dt - nseTimes)
            
            self._curr = self.g * self._s * (vM - self.vE)
            self._draw = False
        elif (self._draw) & (dt == 0): 
            return np.copy(self._curr)
        elif (dt == 0): 
            return np.copy(self._prev_curr)
                    
        return np.copy(self._curr)
        
    # turns on a flag to draw a new random later
    def update(self, dt, spikes, weights):
        """
        Updates the current and kinetic.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 
            
        """
        self._draw = True
        self.record_i(dt)
    
# -------------------- Receptor Currents -----------------------------

class AMPAR(currentBase):
    """
    AMPA receptor mediated current for a neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    g : float
        Conductance of the AMPA receptor in uS.
    vE : float
        The resting potential of these receptors.
    kinetic : kineticBase
        The class that instantiates the AMPA receptor's kinetics.
    
    Methods
    -------
    __init__(size)
        Initializes AMPAR.
    current(dt, vM)
        Gets the currents' value.
    project(dt, vM)
        Predicts future current values.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
    
    """
    def __init__(self, size, order=1, g=0.7, vE=0, 
                 alpha_s=None, alpha_x=None, tau_s=None, tau_x=None,
                 **kwargs):
        """
        Initializes the AMPAR class.

        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        order : int, optional
            The order of the kinetic (1 or 2). The default is 1.
        g : float, optional
            Conductance of the AMPA receptor in uS. The default is 0.7.
        vE : float, optional
             The resting potential of these receptors. The default is 0.
        alpha_s : float, optional
            The rate of increase of the s kinetic. The default is 1.
        alpha_x : TYPE, optional
            The rate of increase of the x kinetic. The default is None if 
            the kinetic's order is 1 and 1 if the order is 2.
        tau_s : TYPE, optional
            The average life expectancy of the s kinetic. The default is
            2 ms if the kinetic's order is 1 and 0.05 ms if the order is 2.
        tau_x : TYPE, optional
            The average life expectancy of the x kinetic. The default is
            None if the kinetic's order is 1 and 2 ms if the order is 2.

        """        
        super().__init__(size, **kwargs)
        
        # Order of the kinetic
        self._order = order                                     
        self.kinetic = None
        self.g = g
        self.vE = vE
        
        # Sets the default AMPA kinetics
        if self._order == 1:
            self.kinetic = firstOrdKinExp(size, 1, 2)
        elif self._order == 2:
            self.kinetic = secondOrdKin(size, 1, 1, 0.05, 2)
        else: 
            raise Exception('Invalid kinetic assigned to AMPACurrent')
        
        if alpha_s is not None: self.kinetic.alpha_s = alpha_s
        if alpha_x is not None: self.kinetic.alpha_x = alpha_x
        if tau_s is not None: self.kinetic.tau_s = tau_s
        if tau_x is not None: self.kinetic.tau_x = tau_x
            
    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self.kinetic.reset()
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """                
        if dt == 0: 
            return np.copy(self._curr)
        else:
            return (self.g * self.kinetic.project(dt) * (vM - self.vE))
        
    def update(self, dt, spikes, weights):
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        """        
        self.kinetic.update(dt, spikes, weights)
        self._prev_curr = np.copy(self._curr)

    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """        
        if dt == 0: 
            return np.copy(self._prev_curr)
        else:
            self._curr = (self.g * self.kinetic.kinetic(dt) * (vM - self.vE))
            self.record_i(dt)
            return np.copy(self._curr)
         
class NMDAR(currentBase):
    """
    NMDA receptor mediated current for a neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    g : float
        Conductance of the NMDA receptor in uS.
    vE : float
        The resting potential of these receptors in mV.
    mg : float
        Extracellular concentration of magnesium.
    kinetic : kineticBase
        The class that instantiates the NMDA receptor's kinetics.
    
    Methods
    -------
    __init__(size)
        Initializes currentBase.
    current(dt, vM)
        Gets the currents' value.
    project(dt, vM)
        Predicts future current values.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
        
    """
    def __init__(self, size, order=2, g=0.07, vE=0, mg=1,
                 alpha_s=None, alpha_x=None, tau_s=None, tau_x=None,
                 **kwargs):
        """
        Initializes the NMDAR class.

        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        order : int, optional
            The order of the kinetic (1 or 2). The default is 2.
        g : float, optional
            Conductance of the NMDA receptor in uS. The default is 0.07.
        vE : float, optional
            The resting potential of these receptors. The default is 0.
        mg : float, optional
            The extracellular concentration of magnesium. The default is 1.
        alpha_s : float, optional
            The rate of increase of the s kinetic. The default is 1.
        alpha_x : TYPE, optional
            The rate of increase of the x kinetic. The default is None if 
            the kinetic's order is 1 and 1 if the order is 2.
        tau_s : TYPE, optional
            The average life expectancy of the s kinetic. The default is
            80 ms if the kinetic's order is 1 and 2 ms if the order is 2.
        tau_x : TYPE, optional
            The average life expectancy of the x kinetic. The default is
            None if the kinetic's order is 1 and 80 ms if the order is 2.

        """
        super().__init__(size, **kwargs)
        
        # Order of the kinetic
        self._order = order                                      
        self.kinetic = None
        self.g = g                    # Conductance (in uS)
        self.vE = vE                  # Reversal potential (in mV)
        self.mg = mg                  # Extracellular [Mg2+]
            
        if self._order == 1:
            self.kinetic = firstOrdKinExp(size, 1, 80)
        elif self._order == 2:
            self.kinetic = secondOrdKin(size, 1, 1, 2, 80)
        else: 
            raise Exception('Invalid kinetic assigned to NMDAcurrent')
                
        if alpha_s is not None: self.kinetic.alpha_s = alpha_s
        if alpha_x is not None: self.kinetic.alpha_x = alpha_x
        if tau_s is not None: self.kinetic.tau_s = tau_s
        if tau_x is not None: self.kinetic.tau_x = tau_x    
                
    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self.kinetic.reset()
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """                
        if dt == 0: 
            return np.copy(self._curr)
        else:
            return (self.g 
                    * self.kinetic.project(dt) 
                    * (vM - self.vE) 
                    / (1 + self.mg/3.57 * np.exp(-0.062 * vM))
                    )
        
    def update(self, dt, spikes, weights):
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons.

        """        
        self.kinetic.update(dt, spikes, weights)
        self._prev_curr = np.copy(self._curr)

    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """
        
        if dt == 0: 
            return np.copy(self._prev_curr)
        else:
            self._curr = (self.g 
                       * self.kinetic.kinetic(dt) 
                       * (vM - self.vE) 
                       / (1 + self.mg/3.57 * np.exp(-0.062 * vM))
                       )
            self.record_i(dt)
            return np.copy(self._curr)

class GABAR(currentBase):
    """
    GABA receptor mediated current for a neural population.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    g : float
        Conductance of the GABA receptor in uS.
    vI : float
        The resting potential of these receptors.
    kinetic : kineticBase
        The class that instantiates the GABA receptor's kinetics.
    
    Methods
    -------
    __init__(size)
        Initializes currentBase.
    current(dt, vM)
        Gets the currents' value.
    project(dt, vM)
        Predicts future current values.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.

    """    
    def __init__(self, size, order=1, g=0.1, vI=-70, 
                 alpha_s=None, alpha_x=None, tau_s=None, tau_x=None,
                 **kwargs):
        super().__init__(size, **kwargs)
        """
        Initializes the GABAR class.

        Parameters
        ----------
        size : uint
            Size of the neuronal population.
        order : int, optional
            The order of the kinetic (1 or 2). The default is 1.
        g : float, optional
            Conductance of the GABA receptor in uS. The default is 0.1.
        vI : float, optional
             The resting potential of these receptors. The default is 
             -70 mV.
        alpha_s : float, optional
            The rate of increase of the s kinetic. The default is 1.
        alpha_x : TYPE, optional
            The rate of increase of the x kinetic. The default is None if 
            the kinetic's order is 1 and 1 if the order is 2.
        tau_s : TYPE, optional
            The average life expectancy of the s kinetic. The default is
            10 ms if the kinetic's order is 1 and 0.25 ms if the order is 2.
        tau_x : TYPE, optional
            The average life expectancy of the x kinetic. The default is
            None if the kinetic's order is 1 and 10 ms if the order is 2.

        """        
        # Order of the kinetic
        self._order = order                                  
        self.kinetic = False
        self.g = g                                         # Conductance (in uS)
        self.vI = vI                                        # Reversal potential (in mV)
            
        if self._order == 1:
            self.kinetic = firstOrdKinExp(size, 1, 10)
        elif self._order == 2:
            self.kinetic = secondOrdKin(size, 1, 1, 0.25, 10)
        else: 
            raise Exception('Invalid kinetic assigned to NMDAcurrent')
            
        if alpha_s is not None: self.kinetic.alpha_s = alpha_s
        if alpha_x is not None: self.kinetic.alpha_x = alpha_x
        if tau_s is not None: self.kinetic.tau_s = tau_s
        if tau_x is not None: self.kinetic.tau_x = tau_x   
            
    # reset the current and kinetics    
    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self.kinetic.reset()
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """                   
        if dt == 0: 
            return np.copy(self._curr)
        else:
            return (self.g * self.kinetic.project(dt) * (vM - self.vI))
        
    # updates the GABA receptor kinetics
    def update(self, dt, spikes, weights):
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        """
        self.kinetic.update(dt, spikes, weights)
        self._prev_curr = np.copy(self._curr)
        
    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """        
        if dt == 0: 
            return np.copy(self._prev_curr)
        else:
            self._curr = (self.g * self.kinetic.kinetic(dt) * (vM - self.vI))
            self.record_i(dt)
            return np.copy(self._curr)

    
# -------------------- Afferent Currents -----------------------------
        
class affFlat(currentBase):
    """
    Afferent current that equally affects the entire population 
    of neurons and does not change over time.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    cue_sal : float
        Salience of the cue (current to the population).
    
    Methods
    -------
    __init__(size)
        Initializes leak.
    current(dt, vM)
        Gets the currents' value.
    hide()
        Remove the afferent signal.
    project(dt, vM)
        Predicts future current values.
    stim()
        Turn on the afferent signal.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
        
    """
    def __init__(self, size, cue_sal=0.15, **kwargs):
        '''
        Initializes an afferent current that equally affects the entire
        population of neurons and does not change over time.

        Parameters
        ----------
        size : int
            Size of the neural population.
        cue_sal : float, optional
            Salience of the cue, which controls the current to the
            neurons in the population. The default is 0.15.

        '''        
        super().__init__(size, **kwargs)
        
        # Default current parameters
        self.cue_sal = 0.15
            
        # Internal tracking parameters
        self._on = False
        self._onCurrent = np.ones([size,]) * self.cue_sal
        self._offCurrent = np.zeros([size,])

    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self._on = False
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """        
        if self._on: 
            return np.copy(self._onCurrent)
        else: 
            return np.copy(self._offCurrent)
        
    def update(self, dt, spikes, weights):        
        """
        Updates the current.
        
        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of presynaptic neurons 
            to postsynaptic neurons. 

        """        
        pass

    def current(self, dt, vM):
        """
        Returns the currents at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
            
        Returns
        -------
        currents : numpy.ndarray(float)
            Current at dt.
            
        """        
        if self._on: 
            return np.copy(self._onCurrent)
        else: 
            return np.copy(self._offCurrent)
        
    def stim(self, cue_sal = None):
        '''
        Turn on the afferent signal.

        Parameters
        ----------
        cue_sal : float, optional
            Specify a new cue salience (i.e., current to the neural
            population). The default is None, which means that the
            old salience is used.

        '''
        self._on = True
        if not cue_sal is None: 
            self.cue_sal = float(cue_sal)
            self._onCurrent = -np.ones([self.size,]) * self.cue_sal
    
    def hide(self):
        '''
        Remove the afferent signal.
        
        '''
        self._on = False
        
class affRing(affFlat):
    """
    Afferent current that comes from a direction and has unequal
    effects on the population of neurons. This current is stable.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    cue_sal : float
        Salience of the cue (current to the population).
    
    Methods
    -------
    __init__(size)
        Initializes leak.
    current(dt, vM)
        Gets the currents' value.
    hide()
        Remove the afferent signal.
    project(dt, vM)
        Predicts future current values.
    stim()
        Turn on the afferent signal.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
    
    """
    def __init__(self, size, positions, cue_spec=(2*np.pi/0.4), **kwargs):
        '''
        Initializes affRing
        
        Parameters
        ----------
        size : int
            Size of the neural population.
        positions : numpy.array(float)
            Radial orientations of the neurons in the population.
        cue_spec : float, optional
            Specificity of the cue. The default is (2*np.pi/0.4).

        '''
        super().__init__(size, **kwargs)
    
        self.cue_spec = cue_spec
        self.positions = np.copy(positions)
            
        # the on current is set to none, because it needs to have a direction, 
        #    which can only be provided by the stim command.
        self._onCurrent = None
        
    def stim(self, direction, cue_sal = None, cue_spec = None):
        '''
        Turn on the afferent signal.       

        Parameters
        ----------
        direction : float
            Direction associated with the cue.
        cue_sal : float, optional
            Specify a new cue salience (i.e., current to the neural
            population). The default is None, which means that the
            old salience is used.
        cue_spec : float, optional
            Specificity of the cue. The default is None, which means
            that it will use the previously set value.

        '''
        if not cue_sal is None: self.cue_sal = float(cue_sal)
        if not cue_spec is None: self.cue_spec = float(cue_spec)
        
        self._on = True
        self._onCurrent = -(self.cue_sal 
                            * np.exp(self.cue_spec 
                                     * (np.cos(self.positions - direction)-1)
                                     )
                            )
        
class affFlatPoisson(noisePoisson):
    """
    Afferent current that equally affects the entire population 
    of neurons. This current is unstable via a Poisson process.
    
    Attributes
    ----------
    size : uint
        Size of the neuronal population.
    g : float, optional
        The conductance of the AMPA-mediated current.
    vE : float, optional 
        The resting potential of these receptors.
    alpha : float, optional
        The increase in the kinetic per received spike.
    tau : float, optional
        The kinetic's average life expectancy.
    rate : float, optional
        The rate of external firing.
        
    
    Methods
    -------
    __init__(size)
        Initializes leak.
    current(dt, vM)
        Gets the currents' value.
    hide()
        Remove the afferent signal.
    project(dt, vM)
        Predicts future current values.
    stim()
        Turn on the afferent signal.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
    
    """
    def __init__(self, size, **kwargs):
        '''
        Initializes affFlatPoisson.

        Parameters
        ----------
        size : int
            Size of the neuronal population.
        g : float, optional
            The conductance of the AMPA-mediated current. The default
            is 0.001.
        vE : float, optional 
            The resting potential of these receptors. The default is 0.
        alpha : float, optional
            The increase in the kinetic per received spike. The default
            is 1.
        tau : float, optional
            The kinetic's average life expectancy. The default is 2 ms.
        rate : float, optional
            The rate of afferent firing.  The default is 1.8 kHz.

        '''
        kwargs.setdefault('g', 0.001)
        super().__init__(size, **kwargs)
        
        # Default current parameters
        self._on = False

    def reset(self):
        """
        Resets the current and tracking for a new run.
        
        """        
        super().reset()
        self._on = False
    
    def project(self, dt, vM):
        """
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.

        Returns
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        """        
        # Creates a new random number if it needs to
        if (self._draw) & (dt != 0):
            self._prev_curr = np.copy(self._curr)
            
            # Will only apply new AMPA current when "on"
            if self._on:
                nseTimes = np.random.rand(self.size) * dt                
                self._decay(nseTimes)
                self._s += (self.alpha 
                            * np.random.poisson(dt * self.rate, self.size))
                self._decay(dt - nseTimes)             
            else:
                self._decay(dt)
            
            self._curr = self.g * self._s * (vM - self.vE)
            self._draw = False
        elif (self._draw) & (dt == 0): 
            return np.copy(self._curr)
        elif (dt == 0): 
            return np.copy(self._prev_curr)
                    
        return np.copy(self._curr)

    def _decay(self, dt):
        # proportion to decay
        p = np.exp(-dt/self.tau)
        self._s *= p
        
    def stim(self, rate = None, g=None):
        '''
        Turn on the afferent signal.     

        Parameters
        ----------
        rate : float, optional
            The rate of external firing. The default is None, which
            means that it will use the previously set value. 
        g : float, optional
            Conductance of the AMPA receptors. The default is None,
            which means that it will use the previously set value.

        '''
        self._on = True
        if not rate is None: 
            self.rate = float(rate)
        if not g is None: 
            self.g = float(g)
       
    def hide(self):
        '''
        Remove the afferent signal.
        
        '''
        self._on = False

class affRingPoisson(affFlatPoisson):
    """
    Afferent current that unequally affects the entire population 
    of neurons. The current is higher in a certain direction and 
    is unstable via a Poisson process.
    
    Attributes
    ----------
    size : uint
        Size of the neuronal population.
    g : float, optional
        The conductance of the AMPA-mediated current.
    vE : float, optional 
        The resting potential of these receptors.
    alpha : float, optional
        The increase in the kinetic per received spike.
    tau : float, optional
        The kinetic's average life expectancy.
    rate : float, optional
        The rate of afferent firing.        
    
    Methods
    -------
    __init__(size)
        Initializes leak.
    current(dt, vM)
        Gets the currents' value.
    hide()
        Remove the afferent signal.
    project(dt, vM)
        Predicts future current values.
    stim()
        Turn on the afferent signal.
    reset()
        Resets the currents and tracking.
    update(dt, spk_times, weights)
        Updates the currents.
    
    """
    def __init__(self, size, positions, max_rate=0, cue_spec=(2*np.pi/0.4),                 
                 **kwargs):
        '''
        Initialize affRingPoisson

        Parameters
        ----------
        size : int
            Size of the neuronal population.
        g : float, optional
            The conductance of the AMPA-mediated current. The default
            is 0.001.
        vE : float, optional 
            The resting potential of these receptors. The default is 0.
        alpha : float, optional
            The increase in the kinetic per received spike. The default
            is 1.
        tau : float, optional
            The kinetic's average life expectancy. The default is 2 ms.
        positions : numpy.array(float)
            Radial orientations of the neurons in the population.
        max_rate : float, optional
            The maximum rate of firing, which is at the cue's position.
            The default is 0.
        cue_spec : float, optional
            Specificity of the cue. The default is (2*np.pi/0.4).

        '''
        super().__init__(size, **kwargs)
               
        # Sets the specificity of the cues
        self._max_rate = max_rate
        self._cue_spec = cue_spec
        self._positions = np.copy(positions)
        
        # Default current parameters
        self._on = False
                
    def stim(self, direction, max_rate = None, cue_spec = None):
        '''
        Turn on the afferent signal.       

        Parameters
        ----------
        direction : float
            Direction associated with the cue.
        max_rate : float, optional
            The maximum rate of firing, which is at the cue's direction.
            The default is None, which means that it will use the 
            previously set value.
        cue_spec : float, optional
            Specificity of the cue. The default is None, which means
            that it will use the previously set value.

        '''
        if max_rate is not None: self._max_rate = float(max_rate)
        if cue_spec is not None: self._cue_spec = float(cue_spec)
        
        self._on = True
        self.rate = (self._max_rate 
                     * np.exp(self._cue_spec 
                              * (np.cos(self._positions - direction)-1)
                              )
                     )    
        
"""
# Stable fixed current that equally affects the entire population
class OscFlatAff(flatPoissonAff):
    '''
    THIS IS NOT CURRENTLY IN USE AND NEEDS TO BE CHECKED
    '''
    
    _oscRateMult = 2 * np.pi / 1000
    
    def __init__(self, size, freq=10, oscProp=0.10, **kwargs):
        super().__init__(size, **kwargs)
        
        self.frequency = freq
        self.period = 0
        self.oscProp = oscProp
        self._oscRate = self.rate
        
            
        # Internal tracking parameters
        self._on = False
    
    def project(self, dt, vM):
        '''
        Predicts future currents with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this current was last updated.
        vM : float
            The membrane voltage potential.
Returns
        
        -------
        currents : numpy.ndarray(float)
            Predicted current at dt.
            
        '''
        
        # Creates a new random number if it needs to
        if (self._draw) & (dt != 0):
            self._prev_curr = np.copy(self._curr)
            
            #advance the period
            self.period += dt
            self._oscRate = self.rate * (1 + np.sin(self.period 
                                                    * self.frequency 
                                                    * self._oscRateMult
                                                    )
                                             * self.oscProp
                                        )
            
            # Will only apply new AMPA current when "on"
            if self._on:
                nseTimes = np.random.rand(self.size) * dt                
                self._decay(nseTimes)
                self._s += (self.alpha
                            * np.random.poisson(dt * self._oscRate, self.size))
                self._decay(dt - nseTimes)
            else:
                self._decay(dt)
            
            self._curr = self.g * self._s * (vM - self.vE)
            self._draw = False
        elif (self._draw) & (dt == 0): 
            return np.copy(self._curr)
        elif (dt == 0): 
            return np.copy(self._prev_curr)
                    
        return np.copy(self._curr)

        
    # present the afferent signal
    def stim(self, rate = None, g=None, oscProp=None):
        super().stim(rate, g)
        
        if not oscProp is None:
            self.oscProp = float(oscProp)
"""