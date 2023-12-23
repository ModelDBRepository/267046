# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:29:39 2019
By Olivia L. Calvin at the University of Minnesota

9 Oct 2019 - The purpose of this code is to implement an agent base class that 
             can be used with tasks. This code will adapt as I wrote more 
             tasks, and the first task that are designed to interact with is 
             the DPX.
             
             There are two guiding principles for how the agent interacts with 
             tasks. The first is that the agent is responsible for recording 
             data about itself, because if this responsibility were the 
             labratory's then it would hinder reusability. Otherwise, the 
             laboratory would have to know the internal mechanisms of whatever 
             agent it is testing. The second is that neither the agent nor 
             task will know how the other represents behaviors and stimuli. 
             Instead a 'dictionary' will be passed between the two that will 
             contain this information and translate between them.             
             
11 Oct 2019 - NOTE: General guideline for the map is that the map is 
              FROM -> TO. For example, cues come from the world so it would be 
              the world's representation in column 1, and the agent 
              representation in column 2. Acts come from the agent, so the
              agent's representation would be in column 1 and the world in 
              column 2.
             
              The act_map will always be one longer than the number of 
              responses. The first element will be 'no response' and the rest
              will actions. Dimensions of act_map are [#responses+1, 0]. For 
              this first agent, the mapping to action is ['O', 'L', 'R'].
             
21 Oct 2019 - This code was not directly changed, but the back code that goes 
              into ringAgent was adjusted to account for multiple sources of 
              pyramidal spikes.
            
22 Nov 2019 - Updated the agent with the new, faster LIF code. The old version 
              is backed up in the scrap file.
              
06 Apr 2020 - Made some updates to the agents so that they are easier to work
              with. Mostly modified the agents to better use kwargs.         
              
18 Sep 2020 - Improved documentation and made the code more in line with
              Python coding standards.
            
"""

import bisect as bi

import numpy as np

from .agentBase import agentBase
from .currents import affRingPoisson
from . import LIF_Pop
from collect import spikeDC

__version__ = "1.0"

class DPX_Agent(agentBase):
    """
    Dual-Ring atttractor agent that I used for the 2021 paper. The
    agent has one ring for perception and a second for memory that
    interact to produce actions.
    
    Attributes
    ----------
    act_map : dict
        The mapping between the agent's action and the task. Provides
        a response that matches the task
    perc_pyr : LIF_Pop.pyrRingPop
        Pyramidal cell population in the perception network.
    perc_int : LIF_Pop.intRingPop
        Interneuron population in the perception network.
    mem_pyr : LIF_Pop.pyrRingPop
        Pyramidal cell population in the memory network.
    mem_int : LIF_Pop.pyrRingPop
        Pyramidal cell population in the memory network.
    perc_pyr_count : int
        Size of the perception network's pyramidal cell population.
    perc_int_count : int
        Size of the perception network's interneuron population.
    mem_pyr_count : int
        Size of the memory network's pyramidal cell population.
    mem_int_count : int
        Size of the memory network's interneuron population.
    resp_options : int
        The number of response options.
    resp_probs : np.array(float)
        The probabilities of engaging the responses.
    choice : int
        The index of the last choice that the agent made.
    resp_tau : float
        The mean life expectancy of the response kinetic.
    softmax_tau : float
        Controls the relative importance of the two response options.
    
    Methods
    -------
    __init__ - Initialize the agent_base class.
    act - Checks to see whether the agent is prepared to act.
    description - Prints a description of the agent to the console.
    full_reset - Resets the agent for the next experiment.
    hide_cue - Remove information about the world.
    present_cue - Receive information from the world.
    process_TStep - Process a time step.
    pull_data - Extracts data from the agent.
    set_act_weight - Set weights from the networks to the response 
                     accumulators.
    set_weights - Sets the ring and matrices weights.
    state_reset - Resets the agent for the next trial.

    """        
    _init_kwargs = {
        'record_i': {'type': bool, 
                  'help': 'Whether current information should be recorded.'},
        'pp_size': {'type': int, 
                    'help': ('The number of pyramidal neurons in the '
                            + 'perception network. [1024 Default]')},
        'pi_size': {'type': int, 
                    'help': ('The number of interneurons in the '  
                            + 'perception network. [256 Default]')},
        'mp_size': {'type': int, 
                    'help': ('The number of pyramidal neurons in the '
                            + 'memory network. [1024 Default]')},
        'mi_size': {'type': int, 
                    'help': ('The number of interneurons in the ' 
                            + 'memory network. [256 Default]')},      
        'aff_rate': {'type': float, 
                  'help': 'The maximum rate of the afferent signal. (in kHz)'},
        'aff_g': {'type': float, 
                  'help': 'The conductance of the afferent signal. (in uS)'}, 
        'ring_p': {'type': float, 
                  'help': 'The p parameter of the ring weighting.'}, 
        'ring_sigma': {'type': float, 
                  'help': 'The sigma parameter of the ring weighting.'},        
        'pp_AMPAg': {'type': float, 
                  'help':('AMPAR conductance of pyramidal neurons in the '
                          + 'perception network. (in uS)')}, 
        'pp_NMDAg': {'type': float, 
                  'help':('NMDAR conductance of pyramidal neurons in the '
                          + 'perception network. (in uS)')}, 
        'pp_GABAg': {'type': float, 
                  'help':('GABAR conductance of pyramidal neurons in the ' 
                          + 'perception network. (in uS)')}, 
        'pi_AMPAg': {'type': float, 
                  'help':('AMPAR conductance of interneurons in the ' 
                          + 'perception network. (in uS)')}, 
        'pi_NMDAg': {'type': float, 
                  'help':('NMDAR conductance of interneurons in the ' 
                          + 'perception network. (in uS)')}, 
        'pi_GABAg': {'type': float, 
                  'help':('GABAR conductance of interneurons in the '
                         + 'perception network. (in uS)')},
        'mp_AMPAg': {'type': float, 
                  'help':('AMPAR conductance of pyramidal neurons in the '
                          + 'memory network. (in uS)')}, 
        'mp_NMDAg': {'type': float, 
                  'help':('NMDAR conductance of pyramidal neurons in the '
                          + 'memory network. (in uS)')}, 
        'mp_GABAg': {'type': float, 
                  'help':('GABAR conductance of pyramidal neurons in the ' 
                          + 'memory network. (in uS)')}, 
        'mi_AMPAg': {'type': float, 
                  'help':('AMPAR conductance of interneurons in the ' 
                          + 'memory network. (in uS)')}, 
        'mi_NMDAg': {'type': float, 
                  'help':('NMDAR conductance of interneurons in the ' 
                          + 'memory network. (in uS)')}, 
        'mi_GABAg': {'type': float, 
                  'help':('GABAR conductance of interneurons in the '
                         + 'memory network. (in uS)')},
        'mult_NMDA': {'type': float, 
                  'help':('Reduce NMDAR conductance network wide.')},
        'softmax_tau': {'type': float, 
                  'help':('Soft max decision making parameter.')},
        'resp_tau': {'type': float, 
                  'help':('Half life of the response kinetic.')},
            }
        
    def __init__(self, act_map={0: 'O', 1: 'L', 2: 'R'}, 
                 pp_size=1024, pi_size=256, mp_size=1024, mi_size=256,
                 ring_p=0.7, ring_sigma=0.05,
                 pp_AMPAg=0.00, pp_NMDAg=0.37, pp_GABAg=1.25,
                 pi_AMPAg=0.00, pi_NMDAg=0.30, pi_GABAg=1.00,
                 mp_AMPAg=0.00, mp_NMDAg=0.37, mp_GABAg=1.25,
                 mi_AMPAg=0.00, mi_NMDAg=0.35, mi_GABAg=1.00,
                 aff_rate = 1.25, aff_g=0.001,
                 softmax_tau=5, resp_tau=80,
                 resp_options=2, mult_NMDA=1, 
                 record_i=False):
        '''
        Initialize a LIF neural network with dual-rings that serve the
        functions of perception and memory.

        Parameters
        ----------
        act_map : dict, optional
            The mapping between the agent's action and the task. 
            The default is:
                0 : 'O'
                1 : 'L'
                2 : 'R'
        pp_size : int, optional
            The number of pyramidal neurons in the perception network. 
            The default is 1024.
        pi_size : int, optional
            The number of interneurons in the perception network. 
            The default is 256.
        mp_size : int, optional
            The number of pyramidal neurons in the memory network. 
            The default is 1024.
        mi_size : int, optional
            The number of interneurons in the memory network. 
            The default is 256.
        ring_p : float, optional
            The p parameter of the ring weighting. The default is 0.7.
        ring_sigma : float, optional
            The sigma parameter of the ring weighting. The default is 0.05.
        pp_AMPAg : float, optional
            AMPAR conductance of pyramidal neurons in the perception network.
            The default is 0.00 uS.
        pp_NMDAg : float, optional
            NMDAR conductance of pyramidal neurons in the perception network.
            The default is 0.37 uS.
        pp_GABAg : float, optional
            GABAR conductance of pyramidal neurons in the perception network.
            The default is 1.25 uS.
        pi_AMPAg : float, optional
            AMPAR conductance of interneurons in the perception network. 
            The default is 0.00 uS.
        pi_NMDAg : float, optional
            NMDAR conductance of interneurons in the perception network. 
            The default is 0.30 uS.
        pi_GABAg : float, optional
            GABAR conductance of interneurons in the perception network. 
            The default is 1.00 uS.
        mp_AMPAg : float, optional
            AMPAR conductance of pyramidal neurons in the memory network.
            The default is 0.00 uS.
        mp_NMDAg : float, optional
            NMDAR conductance of pyramidal neurons in the memory network.
            The default is 0.37 uS.
        mp_GABAg : float, optional
            GABAR conductance of pyramidal neurons in the memory network.
            The default is 1.25 uS.
        mi_AMPAg : float, optional
            AMPAR conductance of interneurons in the memory network.
            The default is 0.00 uS.
        mi_NMDAg : float, optional
            NMDAR conductance of interneurons in the memory network.
            The default is 0.35 uS.
        mi_GABAg : float, optional
            GABAR conductance of interneurons in the memory network.
            The default is 1.00 uS.
        aff_rate : float, optional
            The maximum rate of the afferent signal. The default is 1.25 kHz.
        aff_g : float, optional
           The conductance of the afferent signal. The default is 0.001 uS.
        softmax_tau : float, optional
            Soft max decision making parameter. The default is 5.
        resp_tau : float, optional
            Half life of the response kinetic. The default is 80 ms.
        resp_options : int, optional
            The number of response options. The default is 2.
        mult_NMDA : float, optional
            Multiplier to globally reduce NMDAR conductance. The default is 1.
        record_i : bool, optional
            Whether current information should be recorded. The default is False.

        '''
        super().__init__(act_map)
        
        self.perc_pyr_count = pp_size
        self.perc_int_count = pi_size
        self.mem_pyr_count = mp_size
        self.mem_int_count = mi_size
        
        # Initializes the perception and memory networks with default values.
        self.perc_pyr = LIF_Pop.pyrRingPop(
            self.perc_pyr_count, 
            AMPA_g = pp_AMPAg,
            NMDA_g = pp_NMDAg * mult_NMDA,
            GABA_g = pp_GABAg,
            record_i = record_i
            )
        self.perc_int = LIF_Pop.interRingPop(    
            self.perc_int_count,
            AMPA_g = pi_AMPAg,
            NMDA_g = pi_NMDAg * mult_NMDA, 
            GABA_g = pi_GABAg,
            record_i = record_i
            )
        self.mem_pyr = LIF_Pop.pyrRingPop(   
            self.mem_pyr_count, 
            AMPA_g = mp_AMPAg,
            NMDA_g = mp_NMDAg * mult_NMDA, 
            GABA_g = mp_GABAg, 
            record_i = record_i
            )
        self.mem_int = LIF_Pop.interRingPop(    
            self.mem_int_count, 
            AMPA_g = mi_AMPAg, 
            NMDA_g = mi_NMDAg * mult_NMDA, 
            GABA_g = mi_GABAg, 
            record_i = record_i
            )
        
        # Set the signal current to self.perc_pyr to be an AMPA-Mediated Poisson 
        self.perc_pyr.afferents = affRingPoisson(self.perc_pyr.size, 
                                                 self.perc_pyr.directions
                                                )
        self.perc_pyr.afferents.stim(0, aff_rate)
        self.perc_pyr.afferents.g = aff_g
        self.perc_pyr.afferents.hide()
        
        # 'Action' neuron kinetics and softmax parameters
        self.resp_options = resp_options
        self.resp_probs = np.zeros([self.resp_options,], dtype = 'float')
        self.choice = 0
        self._resp_kin = np.zeros([self.resp_options,], dtype = 'float')
        self.resp_tau = resp_tau
        self.softmax_tau = softmax_tau
                
        self._record_resp_kin = True
        self._resp_kin_record = np.empty((0,self.resp_options))
        
        
        # Sets the parameters for the ring weights
        self._ring_prop = ring_p
        self._ring_sigma = ring_sigma
        
        # Initialize neuronal intra- and inter- connections
        self._perc_PtoP_wts, self._perc_PtoI_wts = [], [] 
        self._perc_ItoP_wts, self._perc_ItoI_wts = [], []
        self._mem_PtoP_wts, self._mem_PtoI_wts = [], []
        self._mem_ItoP_wts, self._mem_ItoI_wts = [], []
        self._perc_to_mem_wts = []
        self._perc_to_act_wts, self._mem_to_act_wts = [], []
        self.set_weights(self._ring_sigma, self._ring_prop)
              
        # Internal data tracking
        self._rec_pp = spikeDC(self.perc_pyr.size, 'Perception Pyramidals')
        self._rec_pi = spikeDC(self.perc_int.size, 'Perception Interneurons')
        self._rec_mp = spikeDC(self.mem_pyr.size, 'Memory Pyramidals')
        self._rec_mi = spikeDC(self.mem_int.size, 'Memory Interneurons')
    
    def act(self):
        '''
        Checks to see whether the agent is prepared to act.

        Returns
        -------
        object
            Returns chosen action from act_map.

        '''
        # implements softMax with the activity traces and action weights
        self.resp_probs = np.exp(self._resp_kin / self.softmax_tau)
        self.resp_probs /= np.sum(self.resp_probs)
                
        # Draws a random number between 0 and 1
        r = float(np.random.rand(1))
        
        #This will return the default 'O'
        self.choice = 0
        
        # Looks for the correct response    
        for i in (np.arange(self.resp_options)+1):
            if r <= np.sum(self.resp_probs[:i]): 
                self.choice = i
                break
       
        return self.act_map[self.choice]
           
    def present_cue(self, direction, **kwargs): 
        '''
        Receive information from the world.

        Parameters
        ----------
        direction : float
            Radial direction that is associated with the cue.
        
        See currents.directionalPoissonAff for details on other kwargs.

        '''
        self.perc_pyr.stim(direction, **kwargs)
                
    def hide_cue(self): 
        '''
        Remove information about the world.

        '''
        self.perc_pyr.removeStim()
        
    def pull_data(self):
        '''
        Extracts data from the agent. CAN BE IMPROVED.

        Returns
        -------
        list
            List of the data recorders.

        '''
       
        return [self._rec_pp, 
                self._rec_pi, 
                self._rec_mp, 
                self._rec_mi]

    def state_reset(self):
        '''
        Resets the agent for the next trial.

        '''
        self._rec_pp = spikeDC(self.perc_pyr.size, 'Perception Pyramidals')
        self._rec_pi = spikeDC(self.perc_int.size,
                                   'Perception Interneurons')
        self._rec_mp = spikeDC(self.mem_pyr.size, 'Memory Pyramidals')
        self._rec_mi = spikeDC(self.mem_int.size, 'Memory Interneurons')

        # Resets the perception and memory networks to a blank slate
        self.perc_pyr.reset()
        self.mem_pyr.reset()
        self.perc_int.reset()
        self.mem_int.reset()
        
        # Resets the tracked response kinetic
        self._resp_kin_record = np.empty((0,self.resp_options))        
        
    def full_reset(self): 
        '''
        Resets the agent for the next experiment.

        '''
        self.state_reset()


    def set_weights(self, sigma, prop, p_to_m = 0.10):
        '''
        Sets the ring and matrices weights

        Parameters
        ----------
        sigma : float
            Width of the Gaussian-like component of the circle weights.
        prop : float
            Proportion of the weights that is comprised of the 
            Guassian-like component.
        p_to_m : float, optional
            Multiplier of the weights between the perception and memory 
            networks. The default is 0.10.
            
        '''
        def _circ_w(sigma, prop, from_pop, to_pop):
            '''
            Creates circular weights for the ring attractor.

            Parameters
            ----------
            sigma : float
                Width of the Gaussian-like component of the circle weights.
            prop : float
                Proportion of the weights that is comprised of the 
                Guassian-like component.
            from_pop : LIF_Pop
                Neural population that the connections are coming from.
            to_pop : LIF_Pop
                Neural population that the connections going to.

            Returns
            -------
            weights : numpy.array(float)
                Normalized connection weights.

            '''
            # Initialize the differences matrix
            _diffMatrix = (np.ones([to_pop.size, from_pop.size], 
                                   dtype='float')
                           * from_pop.directions)
            _contrast = (np.ones([to_pop.size,], dtype = 'float') 
                         * to_pop.directions).reshape(-1, 1)
            _diffMatrix = _diffMatrix -_contrast
            
            
            # Calculates the weights
            weights = ((1 - prop) 
                        + prop * np.exp(2 * np.pi / sigma 
                                     * (np.cos(_diffMatrix) - 1)
                                     )
                        )
            
            # Normalizes the weights
            weights = weights / np.sum(weights, axis=1)
            weights = weights.T
            
            return weights
            
        def _flat_w(from_pop, to_pop): 
            '''
            Creates flat weights for the ring attractor.            

            Parameters
            ----------
            from_pop : LIF_Pop
                Neural population that the connections are coming from.
            to_pop : LIF_Pop
                Neural population that the connections going to.

            Returns
            -------
            weights : numpy.array(float)
                Normalized connection weights.

            '''
            return (np.ones([from_pop.size, to_pop.size], dtype = 'float') 
                    / from_pop.size)
            
        # Sets the parameters for the ring weights
        self._ring_prop = prop          # difference between the max and min
        self._ring_sigma = sigma  # in radians
        
        # Circular Connections
        self._perc_PtoP_wts = _circ_w(self._ring_sigma, self._ring_prop, 
                                    self.perc_pyr, self.perc_pyr)
        self._mem_PtoP_wts = _circ_w(self._ring_sigma, self._ring_prop, 
                                    self.mem_pyr, self.mem_pyr)
        self._perc_to_mem_wts = _circ_w(self._ring_sigma, self._ring_prop, 
                                    self.perc_pyr, self.mem_pyr)
        self._perc_to_mem_wts = (self._perc_to_mem_wts.T * p_to_m).T
        
        # Flat Connections
        self._perc_PtoI_wts = _flat_w(self.perc_pyr, self.perc_int)
        self._perc_ItoP_wts = _flat_w(self.perc_int, self.perc_pyr)
        self._perc_ItoI_wts = _flat_w(self.perc_int, self.perc_int)
        self._mem_PtoI_wts = _flat_w(self.mem_pyr, self.mem_int)
        self._mem_ItoP_wts = _flat_w(self.mem_int, self.mem_pyr)
        self._mem_ItoI_wts = _flat_w(self.mem_int, self.mem_int)
        
        # Sets the action weights
        self._perc_to_act_wts = np.zeros([self.resp_options, 
                                    self.perc_pyr.size], dtype = 'float')        
        self._mem_to_act_wts = np.zeros([self.resp_options, 
                                   self.mem_pyr.size], dtype = 'float')
        
    def set_act_weight(self, low_dir, high_dir, weight, 
                       from_network, accum,):
        '''
        Set weights from the networks to the response accumulators by 
        defining the bounds, weight, and direction.

        Parameters
        ----------
        low_dir : float
            Direction for the lowest bound that will be weighted.
        high_dir : float
            Direction for the upper bound that will be weighted.
        weight : float
            Weight to be assigned.
        from_network : str
            String specifier for the network that the connections are
            coming from. The two options are 'perc' and 'mem' for the
            perception and memory networks, respectively.
        accum : int
            Index of the response accumulator to be assigned.

        '''
        lBound = 0
        if from_network == 'perc':
            uBound = self.perc_pyr_count
            directions = self.perc_pyr.directions[:]
        elif from_network == 'mem':
            uBound = self.mem_pyr_count
            directions = self.mem_pyr.directions[:]       
        else:
            raise AttributeError("from_network must be either 'perc' or 'mem'.")
            
        lBound = bi.bisect_left(directions,low_dir)
        uBound = bi.bisect(directions,high_dir)
        
        if from_network == 'perc':
            self._perc_to_act_wts[accum][lBound:uBound] = weight        
        elif from_network == 'mem':
           self._mem_to_act_wts[accum][lBound:uBound] = weight
        
    def process_TStep(self, dt, time):
        '''
        Processes a time step.

        Parameters
        ----------
        dt : float
            Change in time (ms).
        time : float
            Current time (ms).

        '''
        # Determines which neurons will fire in this time step.        
        self.perc_pyr.predictSpikes(dt, time)
        self.mem_pyr.predictSpikes(dt, time)
        self.perc_int.predictSpikes(dt, time)
        self.mem_int.predictSpikes(dt, time)
        
        # Handle the perception ring's firings
        self.perc_pyr.receivePyrSpikes(dt, 
                                      self.perc_pyr.numSpikes, 
                                      self.perc_pyr.spikeTimes(), 
                                      self._perc_PtoP_wts)
        self.perc_pyr.receiveIntSpikes(dt, 
                                      self.perc_int.numSpikes, 
                                      self.perc_int.spikeTimes(), 
                                      self._perc_ItoP_wts)
        self.perc_int.receivePyrSpikes(dt, 
                                      self.perc_pyr.numSpikes, 
                                      self.perc_pyr.spikeTimes(), 
                                      self._perc_PtoI_wts)
        self.perc_int.receiveIntSpikes(dt, 
                                      self.perc_int.numSpikes, 
                                      self.perc_int.spikeTimes(), 
                                      self._perc_ItoI_wts)
        
        # Pass perception pyramidal spikes to the memory ring
        self.mem_pyr.receivePyrSpikes(dt, 
                                     self.perc_pyr.numSpikes, 
                                     self.perc_pyr.spikeTimes(), 
                                     self._perc_to_mem_wts)        
        
        # Handle the memory networks firings
        self.mem_pyr.receivePyrSpikes(dt, 
                                     self.mem_pyr.numSpikes, 
                                     self.mem_pyr.spikeTimes(), 
                                     self._mem_PtoP_wts)
        self.mem_pyr.receiveIntSpikes(dt, 
                                     self.mem_int.numSpikes, 
                                     self.mem_int.spikeTimes(), 
                                     self._mem_ItoP_wts)
        self.mem_int.receivePyrSpikes(dt, 
                                     self.mem_pyr.numSpikes, 
                                     self.mem_pyr.spikeTimes(), 
                                     self._mem_PtoI_wts)
        self.mem_int.receiveIntSpikes(dt, 
                                     self.mem_int.numSpikes, 
                                     self.mem_int.spikeTimes(), 
                                     self._mem_ItoI_wts)       
             
        # Processes the time step
        self.perc_pyr.processTimeStep(dt, time)
        self.mem_pyr.processTimeStep(dt, time)
        self.perc_int.processTimeStep(dt, time)
        self.mem_int.processTimeStep(dt, time)
        
        # Collect spike data
        self._rec_pp.collect(self.perc_pyr.spikes(), time)
        self._rec_pi.collect(self.perc_int.spikes(), time)
        self._rec_mp.collect(self.mem_pyr.spikes(), time)
        self._rec_mi.collect(self.mem_int.spikes(), time)
               
        # Handles the response kinetics
        self._resp_kin += (self._perc_to_act_wts @ self.perc_pyr.spikes() 
                            + self._mem_to_act_wts @ self.mem_pyr.spikes())
        self._resp_kin *= np.exp(-dt/self.resp_tau)
        
        if self._record_resp_kin: 
            self._resp_kin_record = np.vstack((self._resp_kin_record, 
                                              self._resp_kin))
        
    def description(self, return_str=False):
        '''
        Prints a description of the agent to the console.

        Parameters
        ----------
        return_str : bool, optional
            Whether a string should be return rather than printing to
            the console. The default is False.

        Returns
        -------
        desc : string
            Description of the agent (only if return_str is True).

        '''
        def _pop_desc(pop, ins="    ", afferent=False):
            pop_desc = (ins + "Size: " + str(pop.size) + "\n"
                       + ins + "AMPAR g: " + str(pop.AMPA.g) + " uS\n"
                       + ins + "NMDAR g: " + str(pop.NMDA.g) + " uS\n"
                       + ins + "GABAR g: " + str(pop.GABA.g) + " uS\n"
                       + ins + "Leak g: " + str(pop.leak.g) + " uS\n"
                       + ins + "Noise g: " + str(pop.noise.g) + " uS\n"
                       + ins + "Noise rate: " + str(pop.noise.rate) + " kHz\n"                       
                        ) 
            if afferent:
                pop_desc += ins + "Afferent g: " + str(pop.afferents.g) + " uS\n"
                pop_desc += (ins + "Afferent max rate: " + str(pop.noise.rate) 
                             + " kHz\n")
                
            return pop_desc
        wdth = 42
        ins = " " * 2
        pop_ins = " " * 4
        
        desc = ("-" * wdth + "\n"
                + "Dual-Ring LIF Agent".center(wdth) + "\n"
                + ("Module Version " + __version__).center(wdth) + '\n'
                + "-" * wdth + "\n"
                + "Decision Making (SoftMax)\n"
                + ins + "Accumulator Tau: " + str(self.resp_tau) + "\n"
                + ins + "Softmax Tau: " + str(self.softmax_tau) + "\n"
                + "Ring Weighting\n"
                + ins + "Gauss-like Prop.: " + str(self._ring_prop) + "\n"
                + ins + "Gauss-like Sigma.: " + str(self._ring_sigma) + "\n"
                + "Action Mapping\n"
                )
        
        for par, val in self.act_map.items(): 
            desc += (ins + "Response " + str(round(par,3)) 
                    + " action: " + val + "\n")
                
        desc += ("Perception Network\n"
                 + ins + "Pyramidals\n"
                 + _pop_desc(self.perc_pyr, pop_ins, True)
                 + ins + "Interneurons\n"
                 + _pop_desc(self.perc_int, pop_ins)
                 + "Memory Network\n"
                 + ins + "Pyramidals\n"
                 + _pop_desc(self.mem_pyr, pop_ins)
                 + ins + "Interneurons\n"
                 + _pop_desc(self.mem_int, pop_ins)
                     )            
        desc += "-" * wdth + "\n"
            
        if return_str:
            return desc
        else:
            print(desc)

class CueProbe_Agent(agentBase):
    """
    Single-Ring atttractor agent that I used for some cue-probe experiments.
    
    Attributes
    ----------
    act_map : dict
        The mapping between the agent's action and the task. Provides
        a response that matches the task
    perc_pyr : LIF_Pop.pyrRingPop
        Pyramidal cell population in the network.
    perc_int : LIF_Pop.intRingPop
        Interneuron population in the network.
    perc_pyr_count : int
        Size of the network's pyramidal cell population.
    perc_int_count : int
        Size of the network's interneuron population.
        
    Methods
    -------
    __init__ - Initialize the agent_base class.    
    description - Prints a description of the agent to the console.
    full_reset - Resets the agent for the next experiment.
    hide_cue - Remove information about the world.
    present_cue - Receive information from the world.
    process_TStep - Process a time step.
    pull_data - Extracts data from the agent.
    set_weights - Sets the ring and matrices weights.
    state_reset - Resets the agent for the next trial.

    """ 
    
    _init_kwargs = {
        'record_i': {'type': bool, 
                     'help': 'Whether current information should be recorded.'},
        'p_size': {'type': int, 
                   'help': 'The number of pyramidal neurons.'},
        'i_size': {'type': int, 
                   'help': 'The number of interneurons.'},        
        'aff_rate': {'type': float, 
                     'help': 'The maximum rate of the afferent signal. (in kHz)'},
        'aff_g': {'type': float, 
                  'help': 'The conductance of the afferent signal. (in uS)'}, 
        'ring_p': {'type': float, 
                  'help': 'The p parameter of the ring weighting.'}, 
        'ring_sigma': {'type': float, 
                       'help': 'The sigma parameter of the ring weighting (in radians).'},        
        'p_AMPAg': {'type': float, 
                    'help':' AMPA conductance of pyramidal neurons. (in uS)'}, 
        'p_NMDAg': {'type': float, 
                    'help':' NMDA conductance of pyramidal neurons. (in uS)'}, 
        'p_GABAg': {'type': float, 
                    'help':' GABA conductance of pyramidal neurons. (in uS)'}, 
        'i_AMPAg': {'type': float, 
                    'help':' AMPA conductance of interneurons. (in uS)'}, 
        'i_NMDAg': {'type': float, 
                    'help':' NMDA conductance of interneurons. (in uS)'}, 
        'i_GABAg': {'type': float, 
                    'help':' GABA conductance of interneurons. (in uS)'},
            }

   
    def __init__(self, act_map={}, 
                 p_size=1024, i_size=256,
                 ring_p=0.7, ring_sigma=0.05,
                 p_AMPAg=0.00, p_NMDAg=0.35, p_GABAg=1.25,
                 i_AMPAg=0.66, i_NMDAg=0.20, i_GABAg=1.00,
                 aff_rate = 1.25, aff_g=0.001,
                 record_i=False):
        '''
        Initialize a single-ring LIF neural network.

        Parameters
        ----------
        act_map : dict, optional
            The mapping between the agent's action and the task. 
            The default is {} because this agent doesn't engage in actions.
        p_size : int, optional
           The number of pyramidal neurons in the network.
           The default is 1024.
        i_size : int, optional
            The number of interneurons in the network. The default is 256.
        ring_p : float, optional
            The p parameter of the ring weighting. The default is 0.7.
        ring_sigma : float, optional
            The sigma parameter of the ring weighting. The default is 0.05.
        p_AMPAg : float, optional
            AMPAR conductance of pyramidal neurons. The default is 0.00 uS.
        p_NMDAg : float, optional
            NMDAR conductance of pyramidal neurons. The default is 0.35 uS.
        p_GABAg : float, optional
            GABAR conductance of pyramidal neurons. The default is 1.25 uS.
        i_AMPAg : float, optional
            AMPAR conductance of interneurons. The default is 0.66 uS.
        i_NMDAg : float, optional
            NMDAR conductance of interneurons. The default is 0.20 uS.
        i_GABAg : float, optional
            GABAR conductance of interneurons. The default is 1.00 uS.
        aff_rate : float, optional
            The maximum rate of the afferent signal. The default is 1.25 kHz.
        aff_g : float, optional
           The conductance of the afferent signal. The default is 0.001 uS.
        record_i : bool, optional
            Whether current information should be recorded. 
            The default is False.
            
        '''
        super().__init__(act_map)
        
        self.perc_pyr_count = p_size
        self.perc_int_count = i_size
        
        # Initializes the perception and memory networks with default values.
        self.perc_pyr = LIF_Pop.pyrRingPop(self.perc_pyr_count, 
                                           AMPA_g = p_AMPAg, 
                                           NMDA_g = p_NMDAg, 
                                           GABA_g = p_GABAg, 
                                           record_i = record_i
                                           )
        self.perc_int = LIF_Pop.interRingPop(self.perc_int_count, 
                                             AMPA_g = i_AMPAg, 
                                             NMDA_g = i_NMDAg, 
                                             GABA_g = i_GABAg, 
                                             record_i = record_i
                                             )
        
        # Set the signal current to self.perc_pyr to be an AMPA-Mediated Poisson 
        self.perc_pyr.afferents =  affRingPoisson(self.perc_pyr.size, 
                                                  self.perc_pyr.directions
                                                  )        
        self.perc_pyr.afferents.stim(0, max_rate=aff_rate)
        self.perc_pyr.afferents.g = aff_g
        self.perc_pyr.afferents.hide()
        
        # Sets the parameters for the ring weights
        self._ring_prop = ring_p
        self._ring_sigma = ring_sigma
        
        # Create the connections between neurons 
        self._perc_PtoP_wts, self._perc_PtoI_wts = [], []
        self._perc_ItoP_wts, self._perc_ItoI_wts = [], []
        self.set_weights(self._ring_sigma, self._ring_prop)
              
        # Internal data tracking
        self._trackPyr = spikeDC(self.perc_pyr.size, 'Pyramidals')
        self._trackInt = spikeDC(self.perc_int.size, 'Interneurons')
       
    def present_cue(self, direction, **kwargs): 
        '''
        Receive information from the world.

        Parameters
        ----------
        direction : float
            Radial direction that is associated with the cue.
        
        See currents.directionalPoissonAff for details on other kwargs.

        '''
        self.perc_pyr.stim(direction, **kwargs)
                
    def hide_cue(self): 
        '''
        Remove information about the world.

        '''
        self.perc_pyr.removeStim()

    def pull_data(self):
        '''
        Extracts data from the agent. CAN BE IMPROVED.

        Returns
        -------
        list
            List of the data recorders.

        '''
        return [self._trackPyr, 
                self._trackInt]


    # resets the agent for the next trial
    def state_reset(self):
        '''
        Resets the agent for the next trial.

        '''
        self._trackPyr = spikeDC(self.perc_pyr.size, 'Pyramidals')
        self._trackInt = spikeDC(self.perc_int.size, 'Interneurons')

        # Resets the perception and memory networks to a blank slate
        self.perc_pyr.reset()
        self.perc_int.reset()
        
        
    # resets the agent for the next experiment
    def full_reset(self): 
        '''
        Resets the agent for the next experiment.

        '''
        self.stateReset()


    # Sets the ring weights and adjusts the matrices weights to account for the changes
    def set_weights(self, sigma, prop):
        '''
        Sets the ring and matrices weights

        Parameters
        ----------
        sigma : float
            Width of the Gaussian-like component of the circle weights.
        prop : float
            Proportion of the weights that is comprised of the 
            Guassian-like component.
            
        '''
        # Sets the parameters for the ring weights
        self._ring_prop = prop          # difference between the max and min
        self._ring_sigma = sigma  # in radians
        
        # Directional pyramidal interconnections
        self._perc_PtoP_wts = np.ndarray([self.perc_pyr.size, self.perc_pyr.size], 
                                       dtype = 'float')
        diffMatrix = np.ones([self.perc_pyr.size, self.perc_pyr.size], 
                             dtype = 'float') * self.perc_pyr.directions
        diffMatrix = diffMatrix - np.transpose(diffMatrix)
        self._perc_PtoP_wts = ((1 - self._ring_prop) 
                             + self._ring_prop 
                             * np.exp(2 * np.pi 
                                      / self._ring_sigma 
                                      * (np.cos(diffMatrix) - 1)
                                      )
                             )
        self._perc_PtoP_wts = (self._perc_PtoP_wts 
                             / (np.sum(self._perc_PtoP_wts, axis = 1)))
        self._perc_PtoP_wts = self._perc_PtoP_wts.T

        # Create the other perception network weights
        self._perc_PtoI_wts = np.ones([self.perc_int.size, self.perc_pyr.size], 
                                    dtype = 'float').T / self.perc_pyr.size
        self._perc_ItoP_wts = np.ones([self.perc_pyr.size, self.perc_int.size], 
                                    dtype = 'float').T / self.perc_int.size   
        self._perc_ItoI_wts = np.ones([self.perc_int.size, self.perc_int.size],
                                    dtype = 'float').T / self.perc_int.size  
           
        
    # Processes this time step
    def process_TStep(self, dt, time):
        '''
        Processes a time step.

        Parameters
        ----------
        dt : float
            Change in time (ms).
        time : float
            Current time (ms).

        '''
        # Determines which neurons will fire in this time step.        
        self.perc_pyr.predictSpikes(dt, time)
        self.perc_int.predictSpikes(dt, time)
        
        # Handle the perception ring's firings
        self.perc_pyr.receivePyrSpikes(dt, self.perc_pyr.numSpikes, 
                                      self.perc_pyr.spikeTimes(), 
                                      self._perc_PtoP_wts
                                      )
        self.perc_pyr.receiveIntSpikes(dt, self.perc_int.numSpikes, 
                                      self.perc_int.spikeTimes(), 
                                      self._perc_ItoP_wts
                                      )
        self.perc_int.receivePyrSpikes(dt, self.perc_pyr.numSpikes, 
                                      self.perc_pyr.spikeTimes(), 
                                      self._perc_PtoI_wts
                                      )
        self.perc_int.receiveIntSpikes(dt, self.perc_int.numSpikes, 
                                      self.perc_int.spikeTimes(), 
                                      self._perc_ItoI_wts
                                      )

        # Processes the time step
        self.perc_pyr.processTimeStep(dt, time)
        self.perc_int.processTimeStep(dt, time)
        
        # Collect spike data
        self._trackPyr.collect(self.perc_pyr.spikes(), time)
        self._trackInt.collect(self.perc_int.spikes(), time)


    # Print a description of this agent.
    def description(self, return_str=False):
        '''
        Prints a description of the agent to the console.

        Parameters
        ----------
        return_str : bool, optional
            Whether a string should be return rather than printing to
            the console. The default is False.

        Returns
        -------
        desc : string
            Description of the agent (only if return_str is True).

        '''
        def _pop_desc(pop, ins="    ", afferent=False):
            pop_desc = (ins + "Size: " + str(pop.size) + "\n"
                       + ins + "AMPAR g: " + str(pop.AMPA.g) + " uS\n"
                       + ins + "NMDAR g: " + str(pop.NMDA.g) + " uS\n"
                       + ins + "GABAR g: " + str(pop.GABA.g) + " uS\n"
                       + ins + "Leak g: " + str(pop.leak.g) + " uS\n"
                       + ins + "Noise g: " + str(pop.noise.g) + " uS\n"
                       + ins + "Noise rate: " + str(pop.noise.rate) + " kHz\n"                       
                        ) 
            if afferent:
                pop_desc += ins + "Afferent g: " + str(pop.afferents.g) + " uS\n"
                pop_desc += (ins + "Afferent max rate: " + str(pop.noise.rate) 
                             + " kHz\n")
                
            return pop_desc
        wdth = 42
        ins = " " * 2
        
        desc = ("-" * wdth + "\n"
                + "Single-Ring LIF Agent".center(wdth) + "\n"
                + ("Module Version " + __version__).center(wdth) + '\n'
                + "-" * wdth + "\n"
                + "Ring Weighting\n"
                + ins + "Gauss-like Prop.: " + str(self._ring_prop) + "\n"
                + ins + "Gauss-like Sigma.: " + str(self._ring_sigma) + "\n"
                + "Action Mapping\n"
                )
        
        for par, val in self.act_map.items(): 
            desc += (ins + "Response " + str(round(par,3)) 
                    + " action: " + val + "\n")
                
        desc += ("Pyramidals\n"
                 + _pop_desc(self.perc_pyr, ins, True)
                 + "Interneurons\n"
                 + _pop_desc(self.perc_int, ins)
                     )            
        desc += "-" * wdth + "\n"
            
        if return_str:
            return desc
        else:
            print(desc)