# -*- coding: utf-8 -*-
"""
Purpose
-------
This code implements receptor kinetics for realistic spiking neuron models.

Classes
-------
kineticBase() - Kinetic base class
    firstOrdKin(kineticBase) - First-order kinetic that implements
                               Equation 6 from [1].
    firstOrdKinExp(firstOrdKin) - First-order kinetic that implements
                                  an exponential decay variant from 
                                  [2].
    secondOrdKin(kineticBase) - Second-order kinetic that implments
                                Equations 4 and 5 from [1].

Functions
---------
None

Change log
----------     
03 May 20 - Cleaned up the code to be more Pythonic and added more
            documentation. Final version for eventual upload.
            
18 Sep 20 - Split out the kinetics from the currents.
            
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

from math import exp

from numpy import zeros, copy

class kineticBase(object):
    """
    Base class for kinetics.  This assumes that there is a single
    s kinetic, which can be expanded upon by subclasses.
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    track : boolean
        Whether one of the kinetics should be tracked.
    track_ID : int
        The index of the neuron whose kinetic is being tracked.
    tracked_s : list(float)
        If track is True, the s kinetic value during the experiment.
    
    Methods
    -------
    __init__(size, track=False, track_ID=0)
        Initializes kineticBase
    project(dt)
        Predicts future channel gating.
    update(dt, spk_times, weights)
        Updates the kinetics.
    kinetic(dt)
        Gets the kinetic value.
    reset()
        Resets the kinetics and tracking.
    
    """
    def __init__(self, size, track=False, track_ID=0):
        """
        Initializes kineticBase.

        Parameters
        ----------
        size : int
            Size of the neural population
        track : boolean
            Whether the kinetics should be tracked. The default is False.
        track_ID : uint
            Which neuron's kinetics should be tracked. The default is 0.
        Returns
        -------
        None.

        """
        
        self.size = size      # size of the population
        
        #tracking variables
        self.track = track
        self.track_ID = track_ID
        if self.track_ID >= self.size:
            raise ValueError('track_ID was set to a value larger than the'
                             + ' size of the neural population.')
        self.tracked_s = []
        
        # S kinetic parameter
        self._s = zeros([self.size,], dtype='float')      # channel gating
        self._prev_s = copy(self._s)

    def project(self, dt):
        """
        Predicts future channel gating with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this kinetic was last updated.

        Raises
        ------
        NotImplementedError
            This function is not implemented in the kineticBase.

        Returns
        -------
        s_kinetic : numpy.ndarray(float)
            Predicted channel gatings at dt.
            
        """
        
        raise NotImplementedError(
            'Trying to call "project" with the kineticBase class')
        
    def update(self, dt, spk_times, weights):
        """
        Updates the kinetics based on the population spiking and the 
        connection weights.

        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of the presynaptic neuron to 
            postsynaptic neurons. 

        Raises
        ------
        NotImplementedError
            This function is not implemented in the kineticBase.

        Returns
        -------
        None.

        """
        
        raise NotImplementedError(
            'Trying to call "update" with the kineticBase class')
        
    def kinetic(self, dt):
        """
        Returns the kinetic value at the beginning or end of the dt.
        Should only be called after update() has been called.

        Parameters
        ----------
        dt : float
            The change in time since this kinetic was last updated.

        Returns
        -------
        s_kinetic : numpy.ndarray(float)
            Channel gating by neuron at dt.
            
        """
        
        # Returns the kinetic value at the start of this time step
        if dt == 0: return copy(self._prev_s)
        return copy(self._s)
    
    def reset(self):
        """
        Resets the s kinetic and tracking for a new run.

        Returns
        -------
        None.
        
        """        
        
        self._s = zeros([self.size,], dtype='float')
        self._prev_s = copy(self._s)
        self.tracked_s = []

# -------------------- First Order Kinetics -----------------------------
        
class firstOrdKin(kineticBase):
    """
    A first-order kinetic that increases by alpha with every spike, 
    is bounded, and proportionately decreases. Implements equation 6 
    from [1]. This is a subclass of kineticBase.
    
    [1] Wang, X.J. (1999). Synaptic Basis of Cortical Persistent 
        Activity:  the Importance of NMDA Receptors to Working Memory. 
        The Journal of Neuroscience, 19(21), 9587–9603. 
        doi: 10.1523/JNEUROSCI.19-21-09587.1999
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    track : boolean
        Whether one of the kinetics should be tracked.
    track_ID : int
        The index of the neuron whose kinetic is being tracked.
    tracked_s : list(float)
        If track is True, the s kinetic value during the experiment.
    alpha_s : float
        Multiplier to increase the s kinetic by each spike.
    tau_s : float
        Average life expectancy of the s kinetic.
    
    Methods
    -------
    __init__(size, alpha, tau, **kwargs)
        Initializes FirstOrdKin
    _loss_prop(dt)
        Determines the lost proportion of the kinetic.
    project(dt)
        Predicts future channel gating.
    update(dt, spk_times, weights)
        Updates the kinetics.
    kinetic(dt)
        Gets the kinetic value.
    reset()
        Resets the kinetics and tracking.
    
    """
    
    def __init__(self, size, alpha_s, tau_s, **kwargs):
        """
        Initalize FirstOrdKin.  

        Parameters
        ----------
        size : int
            Size of the neural population.
        alpha_s : float
            Multiplier to increase the s kinetic by each spike.
        tau_s : float
            Average life expectancy of the s kinetic (in ms).
        **kwargs : dict
            Keyword arguments. Accepts the following kwargs:
                track (boolean) 
                track_ID (int) 

        Returns
        -------
        None.

        """
        
        super().__init__(size, **kwargs)

        self.alpha_s = alpha_s
        self.tau_s = tau_s
    
    def _loss_prop(self, dt): 
        """
        Proportion of the s kinetic that was lost.

        Parameters
        ----------
        dt : float
            Time that the kinetics for.
        tau : float
            Average life expectancy of the kinetic.

        Returns
        -------
        prop : float
            The proportion of the kinetic that was lost.

        """
        return (1 - dt/self.tau_s)
        
    # quickly predicts the channel gating
    def project(self, dt): 
        """
        Predicts future channel gating with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this kinetic was last updated.

        Returns
        -------
        s_kinetic : numpy.ndarray(float)
            Predicted channel gatings at dt.
            
        """
        
        if dt == 0: return copy(self._s)
        
        # proportion after decay
        return self._s * self._loss_prop(dt)
    
    # updates the kinetic for this time step
    def update(self, dt, spk_times, weights):
        """
        Updates the kinetics based on the population spiking and the 
        connection weights.

        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of the presynaptic neuron to 
            postsynaptic neurons. 

        Returns
        -------
        None.

        """          
        
        _time = 0
        self._prev_s = copy(self._s)
        
        # Processes Equation 6 in a stepwise manner.
        for index, spk_t in enumerate(spk_times):
            self._s *= self._loss_prop(spk_t - _time)
            self._s += self.alpha_s * weights[index] * (1 - self._s)
            _time = spk_t
            
        # Decays the rest of the time.
        self._s *= self._loss_prop(dt - _time)
        
        # tracks the kinetic of a single neuron if instructed to do so
        if self.track: self.tracked_s.append(self._s[self.track_ID])

class firstOrdKinExp(firstOrdKin):
    """
    A first-order kinetic that increases by alpha with every spike, 
    is unbounded, and exponentially decreases [2]. This is a subclass 
    of FirstOrdKin.
    
    [2] Compte, A., Brunel, N., Goldman-Rakic, P.S., & Wang, X.J. (2000). 
        Synaptic mechanisms and network dynamics underlying spatial 
        working memory in a cortical network model. Cerebral Cortex, 10,
        910-923. doi: 10.1093/cercor/10.9.910
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    track : boolean
        Whether one of the kinetics should be tracked.
    track_ID : int
        The index of the neuron whose kinetic is being tracked.
    tracked_s : list(float)
        If track is True, the s kinetic value during the experiment.
    alpha_s : float
        Multiplier to increase the s kinetic by each spike.
    tau_s : float
        Average life expectancy of the s kinetic.
    
    Methods
    -------
    __init__(size, alpha, tau, **kwargs)
        Initializes FirstOrdKinExp
    _loss_prop(dt)
        Determines the lost proportion of the kinetic.
    project(dt)
        Predicts future channel gating.
    update(dt, spk_times, weights)
        Updates the kinetics.
    kinetic(dt)
        Gets the kinetic value.
    reset()
        Resets the kinetics and tracking.
    
    """
    
    def __init__(self, size, alpha_s, tau_s, **kwargs):
        """
        Initalize FirstOrdKinExp.  

        Parameters
        ----------
        size : int
            Size of the neural population.
        alpha_s : float
            Multiplier to increase the s kinetic by each spike.
        tau_s : float
            Average life expectancy of the s kinetic (in ms).
        **kwargs : dict
            Keyword arguments. Accepts the following kwargs:
                track (boolean) 
                track_ID (int) 

        Returns
        -------
        None.

        """
        super().__init__(size, alpha_s, tau_s, **kwargs)
    
    def _loss_prop(self, dt): 
        """
        Proportion of the s kinetic that was lost.

        Parameters
        ----------
        dt : float
            Time that the kinetics for.
        tau : float
            Average life expectancy of the kinetic.

        Returns
        -------
        prop : float
            The proportion of the kinetic that was lost.

        """
        return exp(-dt/self.tau_s)
    
    # updates the kinetic for this time step
    def update(self, dt, spk_times, weights):
        """
        Updates the kinetics based on the population spiking and the 
        connection weights.

        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of the presynaptic neuron to 
            postsynaptic neurons. 

        Returns
        -------
        None.

        """          
        
        _time = 0
        self._prev_s = copy(self._s)
        
        for index, spk_t in enumerate(spk_times):
            self._s *= self._loss_prop(spk_t - _time)
            self._s += self.alpha_s * weights[index]                
            _time = spk_t
            
        # Decays the rest of the time.
        self._s *= self._loss_prop(dt - _time)
        
        # tracks the kinetic of a single neuron if instructed to do so.
        if self.track: self.tracked_s.append(self._s[self.track_ID])

class secondOrdKin(kineticBase):
    """
    A second-order kinetic that increases by alpha with every spike, 
    is bounded, and proportionately decreases. Implements equations 
    4 and 5 from [1]. This is a subclass of kineticBase.
    
    [1] Wang, X.J. (1999). Synaptic Basis of Cortical Persistent 
        Activity:  the Importance of NMDA Receptors to Working Memory. 
        The Journal of Neuroscience, 19(21), 9587–9603. 
        doi: 10.1523/JNEUROSCI.19-21-09587.1999
    
    Attributes
    ----------
    size : int
        The size of the neural population.
    track : boolean
        Whether one of the kinetics should be tracked.
    track_ID : int
        The index of the neuron whose kinetic is being tracked.
    tracked_s : list(float)
        If track is True, the s kinetic value during the experiment.
    alpha_x : float
        Multiplier to increase the x kinetic by each spike.
    alpha_s : float
        Rate that the x kinetic influences the s kinetic.
    tau_x : float
        Average life expectancy of the x kinetic (in ms).
    tau_s : float
        Average life expectancy of the s kinetic (in ms).
    
    Methods
    -------
    __init__(size, alpha_x, alpha_s, tau_x, tau_s **kwargs)
        Initializes SecondOrdKin
    _dsdt(dt)
        Change in the s kinetic over time.
    project(dt)
        Predicts future channel gating.
    update(dt, spk_times, weights)
        Updates the kinetics.
    kinetic(dt)
        Gets the kinetic value.
    reset()
        Resets the kinetics and tracking.
    
    """
    
    def __init__(self, size, alpha_x, alpha_s, tau_x, tau_s, **kwargs):
        """
        Initialize SecondOrdKin.

        Parameters
        ----------
        size : int
            Size of the neural population.
        alpha_x : float
            Multiplier to increase the x kinetic by each spike.
        alpha_s : float
            Rate that the x kinetic influences the s kinetic.
        tau_x : float
            Average life expectancy of the x kinetic (in ms).
        tau_s : float
            Average life expectancy of the s kinetic (in ms).
        **kwargs : dict
            Keyword arguments. Accepts the following kwargs:
                track (boolean) 
                track_ID (int) .

        Returns
        -------
        None.

        """
       
        super().__init__(size, **kwargs)
        
        self.alpha_x = alpha_x                             
        self.tau_x = tau_x                             
        self.alpha_s = alpha_s
        self.tau_s = tau_s
        
        # spike kinetic
        self._x = zeros([self.size,], dtype='float') 
        self._prev_x = copy(self._x)
        
        # tracked values
        self.tracked_x = []
            
    def reset(self):
        """
        Resets the s and x kinetics and tracking for a new run.

        Returns
        -------
        None.
        
        """  
        
        super().reset()
        self._x = zeros([self.size,], dtype='float')
        self._prevX = copy(self._s)
        self.trackedX = []
    
    # project the channel gating into the future
    def project(self, dt):
        """
        Predicts future channel gating with the assumption of no new firing.

        Parameters
        ----------
        dt : float
            The change in time since this kinetic was last updated.

        Returns
        -------
        s_kinetic : numpy.ndarray(float)
            Predicted channel gatings at dt.
            
        """
        if dt == 0: 
            return copy(self._s)
        else: 
            return (self._s + self._dsdt(dt))
    
    def _dsdt(self, dt):
        """
        Change in the s kinetic over time.

        Parameters
        ----------
        dt : float
            Time since the kinetics were last updated.

        Returns
        -------
        dsdt : np.array(float)
            Change in the s kinetic.

        """
        
        return (dt * (self.alpha_s 
                      * self._x 
                      * (1 - dt/self.tau_x) 
                      * (1 - self._s) 
                      - self._s/self.tau_s
                      )
                )
    
    def update(self, dt, spk_times, weights):
        """
        Updates the kinetics based on the population spiking and the 
        connection weights.

        Parameters
        ----------
        dt : float
            The change in time.
        spk_times : numpy.ndarray(float; spikes)
            When spikes occur during this dt.
        weights : numpy.ndarray(float; spikes x size)
            Matrix of connection weights of the presynaptic neuron to 
            postsynaptic neurons. 

        Returns
        -------
        None.

        """   
            
        _time = 0
        _d_dur = 0
        self._prev_x = copy(self._x)
        self._prev_s = copy(self._s)        
        
        for index, spk_t in enumerate(spk_times):
            _d_dur = spk_t - _time
            self._s += self._dsdt(_d_dur)
            self._x *= (1 - _d_dur/self.tau_x)
            self._x += self.alpha_x * weights[index]
            _time = spk_t
            
        # Decays the rest of the time.
        _d_dur = dt - _time
        self._s += self._dsdt(_d_dur)
        self._x *= (1 - _d_dur/self.tau_x)
        
        # tracks the kinetics of a single neuron if instructed to do so.
        if self.track: 
            self.tracked_s.append(self._s[self.track_ID])
            self.tracked_x.append(self._x[self.track_ID])