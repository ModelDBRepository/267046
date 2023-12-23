# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14
By Olivia L. Calvin at the University of Minnesota

2020 March 12 - Updated the classes to better reflect good Python practices
                and added some slight updates to the DPX class.
"""

from os.path import isfile

import numpy as np
from numpy import random as rand

from .baseWorld import BaseWorld

class DPX(BaseWorld):
    """
    A task that is used to assess context integration.
    
    Attributes
    ----------
    cueMap : dict
        How cues are mapped to the agent.
    dt : float
        How quickly time changes.
    trials : int
        The number of trials
    aProp : float
        The proportion of trials with an 'A' cue.
    axProp : float
        The proportion of trials with an 'A' cue paired with an 'X' probe.
    bxProp : float
        The proportion of trials with a 'B' cue paired with an 'X' probe.
    preCueDur : float
        The time before the cue starts during each trial.
    cueDur : float
        The duration of the cue.
    ISI : float
        The duration between the cue and probe.
    probeDur : float
        The duration of the probe.
    ITI : float
        The time after the probe until the end of the trial.
    actTime : float
        When the agent's action is probed.
    rewardV : float
        The value of the reward sent to the agent.
    failV : float
        The value of a failure to the agent.
    trials : list
        The trial types.
    numAX : int
        The number of 'AX' trials.        
    numBX : int
        The number of 'BX' trials.
    numAY : int
        The number of 'AY' trials.
    numBY : int
        The number of 'BY' trials.
            
    Methods
    -------
    reset()    
        Resets the DPX task so that it is ready for another agent.
    set_trials(numTrials, aProp, axProp, bxProp)
        Prepares the DPX by allocating the trials to various cue and probe
        combinations in proportion to the number trials associated with
        each.
    set_cues(aCue, *bCues)
        Sets the cues and recreates the trials based on them.
    set_probes(xProbe, *yProbes)
        Sets the probes and recreates the trials based on them.
    current_cp()
        Returns the current trial's cue and probe.
    next_trial(agent)
        Conducts the next DPX trial. Returns -1 when all trials have been
        conducted.
    run_trial(agent, cue, probe, target)
        Runs a trial of the DPX using the specified cue and probe.
    output_data(save_loc, trial=None)
        Output the behavioral data. Can either send the entire data or one trial.
    description()
        Prints to screen a description of the task.
"""
    
    __version__ = '1.2'
    
    # Output format
    _out_frmt = ['%.8s', '%.8s', '%.8s', '%.8s', '%.8s']
    _out_header = 'trial,cue,probe,correct,action'
    
    # Arguments that can be used by an argument parser.
    _kwargs_list = {
        'dt': {'type': float, 'help': 'Size of the time steps.'},
        'trials': {'type': int, 'help': 'The number of trials to conduct.'},
        'aProp': {'type': float, 
                  'help': 'The proportion of trials that use the A cue.'},
        'axProp': {'type': float, 
                  'help': 'The proportion of A cue trials that have X probes.'},
        'bxProp': {'type': float, 
                  'help': 'The proportion of B cue trials that have X probes.'}, 
        'dPre': {'type': int, 
                  'help': 'Duration of time prior to the cue. (in ms)'}, 
        'dCue': {'type': int, 
                  'help': 'Duration of cue presentation. (in ms)'}, 
        'dISI': {'type': int, 
                  'help': 'Duration of the interstimulus interval. (in ms)'}, 
        'dProbe': {'type': int, 
                  'help': 'Duration of probe presentation. (in ms)'}, 
        'dITI': {'type': int, 
                  'help': 'Duration of the intertrial interval. (in ms)'}, 
        'actTime': {'type': int, 
                  'help': 'When the agent can start to act. (in ms)'},  
        'rewardV': {'type': float, 'help': 'Value of the reward'},  
        'failV': {'type': float, 'help': 'Value of a failure'},
            }
    
    def __init__(self, cueMap, trials=150, aProp=0.8, axProp=0.8, bxProp=0.8,
                 dPre=500, dCue=1000, dISI=4000, dProbe=500, dITI=600, 
                 actTime=6000, rewardV=1, failV=0, **kwargs):
        super().__init__(cueMap, **kwargs)
        
        self.numTrials = int(trials)
        self.aProp, self.axProp, self.bxProp = aProp, axProp, bxProp
        self.preCueDur, self.cueDur, self.ISI = dPre, dCue, dISI
        self.probeDur, self.ITI = dProbe, dITI
        self.actTime = actTime
        self.rewardV, self.failV = rewardV, failV
        
        # Verifies that all of the assigned parameters were valid.
        for param, val in kwargs.items():       
            if not param in self._kwargs_list: 
                raise Exception (param, "does not exist in the DPX class")    
        
        # Set up the cues and probes
        self._aCue = np.array(['A'])
        self._bCues = np.array(['B'])#, 'B2', 'B3', 'B4', 'B5'])
        self._xProbe = np.array(['X'])
        self._yProbes = np.array(['Y'])#, 'Y2', 'Y3', 'Y4', 'Y5'])
        self._targResp = np.array(['L'])
        self._ntargResp = np.array(['R'])  

        # Prepare the trials information
        self.trials = []
        self.numAX, self.numAY, self.numBX, self.numBY = 0, 0, 0, 0
        self._AX, self._AY, self._BX, self._BY = [], [], [], []
        self.set_trials(self.numTrials, self.aProp, self.axProp, self.bxProp)
               
        # Tracking Info
        self._currentTrial = -1

        # Agent Tracking
        self.actions = []        
        

    def reset(self):
        '''
        Resets the DPX task so that it is ready for another agent.

        Returns
        -------
        None.

        '''
        # reset data
        self.actions = []
        self._currentTrial = -1
        
        # shuffles the trials
        rand.shuffle(self.trials)


    @staticmethod
    def _samp_wo_rep(need, trials):
        '''
        Randomly sample from trials without replacement.

        Parameters
        ----------
        need : integer
            The number of trials that need sampled (n).
        trials : np.array(N,)
            List of the trials that can be sampled from.

        Returns
        -------
        temp : np.array(n,)
            Random list of the trials.
        '''
        
        if need > 0:
            temp = np.copy(trials)
            for i in range(trials.shape[0] - need): 
                temp = np.delete(temp, 
                                 rand.randint(0,temp.shape[0]), 
                                 axis = 0
                                 )
        
        return temp

    def set_trials(self, numTrials, aProp, axProp, bxProp):
        '''
        Prepares the DPX by allocating the trials to various cue and probe
        combinations in proportion to the number trials associated with
        each.

        Parameters
        ----------
        numTrials : integer
            The number of trials that will be run.
        aProp : float
            The proportion of trials that will have the 'A' cue.
        axProp : flaot
            The proportion of 'A' cue trials that will have an 'X' probe.
        bxProp : float
            The proportion of 'B' cue trials that will have an 'X' probe.

        Returns
        -------
        None.

        '''
        
        # Ensure that the data passed in is of the correct format.
        if not isinstance(numTrials, int):
            raise ValueError('numTrials is not an integer.')
        if aProp < 0 or aProp > 1:
            raise ValueError('aProp is not between zero and one.')
        if axProp < 0 or axProp > 1:
            raise ValueError('axProp is not between zero and one.')
        if bxProp < 0 or bxProp > 1:
            raise ValueError('bxProp is not between zero and one.')
        
        # Set up the trial proportions
        self.numTrials = numTrials
        self.aProp, self.axProp, self.bxProp = aProp, axProp, bxProp
        self.bProp = 1 - self.aProp 
        self.ayProp = 1 - self.axProp
        self.byProp = 1 - self.bxProp
        
        # Trial Parameters
        self.numAX = round(numTrials * self.aProp * self.axProp)
        self.numAY = round(numTrials * self.aProp * self.ayProp)
        self.numBX = round(numTrials * self.bProp * self.bxProp)
        self.numBY = (numTrials 
                      - (self.numAX + self.numAY + self.numBX))

        # Set up the trials and cue/probe info
        self.cues = np.append(self._aCue, self._bCues)
        self.probes = np.append(self._xProbe, self._yProbes)
        
        # Sets up the trial types
        self._AX = np.array(np.meshgrid(self._aCue, 
                                        self._xProbe,  
                                        self._targResp)
                            ).T.reshape(-1,3)
        self._AY = np.array(np.meshgrid(self._aCue, 
                                        self._yProbes, 
                                        self._ntargResp)
                            ).T.reshape(-1,3)
        self._BX = np.array(np.meshgrid(self._bCues, 
                                        self._xProbe, 
                                        self._ntargResp)
                            ).T.reshape(-1,3)
        self._BY  = np.array(np.meshgrid(self._bCues, 
                                         self._yProbes, 
                                         self._ntargResp)
                             ).T.reshape(-1,3)
        
        # Creates the trials array and shuffles
        self.trials = np.zeros([0,3], dtype = '<U2')

        # Adds the AX trials        
        for i in range(int(self.numAX/self._AX.shape[0])): 
            self.trials = np.vstack((self.trials,self._AX))
        if self.numAX % self._AX.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numAX 
                                                       % self._AX.shape[0]), 
                                                      self._AX)
                                    ))
        
        # Adds the AY trials
        for i in range(int(self.numAY/self._AY.shape[0])): 
            self.trials = np.vstack((self.trials,self._AY))
        if self.numAY % self._AY.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numAY
                                                       % self._AY.shape[0]), 
                                                      self._AY)
                                    ))
        # Adds the BX trials    
        for i in range(int(self.numBX/self._BX.shape[0])):  
            self.trials = np.vstack((self.trials,self._BX))
        if self.numBX % self._BX.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numBX
                                                       % self._BX.shape[0]), 
                                                      self._BX)
                                    ))
        # Adds the BY trials
        for i in range(int(self.numBY/self._BY.shape[0])):  
            self.trials = np.vstack((self.trials,self._BY))
        if self.numBY % self._BY.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numBY
                                                       % self._BY.shape[0]), 
                                                      self._BY)
                                    ))
        
        #self.trials = np.array(self.trials).reshape(-1,3)
        # Shuffles the trials and sets the current trial number
        rand.shuffle(self.trials)
        self._currentTrial = -1

    def set_cues(self, aCue, *bCues):
        '''
        Sets the cues and recreates the trials based on that information.

        Parameters
        ----------
        aCue : str
            The string that represents the 'A' cue.
        *bCues : str
            The strings that represent the 'B' cues. 

        Returns
        -------
        None.

        '''
        self._aCue = np.array(aCue)
        self._bCues = np.array(bCues)
        self.set_trials(self.numTrials, self.aProp, self.axProp, self.bxProp)
      
        
    def set_probes(self, xProbe, *yProbes):
        '''
        Sets the probes and recreates the trials based on that information.

        Parameters
        ----------
        xProbe : str
            The string that represents the 'X' cue
        *yProbes : str
            The strings that represent the 'Y' cues. 

        Returns
        -------
        None.

        '''
        self._xProbe = np.array(xProbe)
        self._yProbes = np.array(yProbes)
        self.set_trials(self.numTrials, self.aProp, self.axProp, self.bxProp)        
    
    
    def current_cp(self):
        '''
        Get the current trial's cue and probe.

        Returns
        -------
        string
            The letters of the cue and probe.

        '''
        _trial = ""
        
        if self._currentTrial >= 0:
            _trial = (self.trials[self._currentTrial][0] 
               + self.trials[self._currentTrial][1])
        
        return _trial


    def next_trial(self, agent):
        '''
        Conducts the next DPX trial.

        Parameters
        ----------
        agent : object
            The agent that will be interacting with the DPX.

        Returns
        -------
        int
            Returns -1 when all of the trials have been conducted.
        '''
                        
        if (self._currentTrial + 1) < self.numTrials:
            # advances to the next trial
            self._currentTrial += 1
            _trial = self.trials[self._currentTrial]
        
            action = self.run_trial(agent, *_trial)
            self.actions.append(action)
        else:
            # Let the calling class know that 
            return -1
        
       
    def run_trial(self, agent, cue, probe, target):
        '''
        Runs a trial of the DPX using the specified cue and probe.

        Parameters
        ----------
        agent : object
            The agent that will be interacting with the DPX.
        cue : str
            The cue that will be presented during the trial. Must be
            in the cueMap.
        probe : str
            The probe that will be presented during the trial. Must be
            in the cueMap.
        target : str
            The desired response for this cue-probe pair. 

        Returns
        -------
        action : str
            The agent's action during the trial.
        '''

        # Reset the trial information
        _tDur = 0
        _acted = False
        action = None
        
        # Execute the trial
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.preCueDur, 
                                              _tDur, _acted)
        agent.present_cue(self.cueMap.get(cue))
        action, _acted, _tDur = self._process(agent, self.cueDur, 
                                              _tDur, _acted)
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.ISI, 
                                              _tDur, _acted)        
        agent.present_cue(self.cueMap.get(probe))
        action, _acted, _tDur = self._process(agent, self.probeDur, 
                                              _tDur, _acted) 
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.ITI, 
                                              _tDur, _acted)               
        
        return action
        
    
    def _process(self, agent, duration, time, acted):
        '''
        Processes a phase of the trial.

        Parameters
        ----------
        agent : object
            The agent that is engaged with the DPX.
        duration : int
            How long this phase of the trial is (in ms).
        time : int
            The time at the start of the trial phase(in ms).
        acted : bool
            Whether the agent has acted.

        Returns
        -------
        action, acted, newTime
            action : str
                The agent's action if it exhibited. Otherwise None.
            acted : bool
                Whether the agent has acted
            newTime : int
                The new time (in ms).
        '''
        
        action = None
        _t = time
        
        for d in range(int(duration/self.dt)):                
            _t = d * self.dt + time
            agent.process_TStep(self.dt, _t)
            
            # Checks to see if the agent needs to act.
            if not acted:
                if _t >= self.actTime:
                    action = agent.act()
                    if action is not None:
                        acted = True
            
        return action, acted, (time+duration)   
        
           
    def output_data(self, saveLoc, trial=None):
        '''
        Output the behavioral data. Can either send everything or one trial

        Parameters
        ----------
        saveLoc : str
            The location that the data needs written to.
        trial : int, optional
            Outputs the data for a specified trial. The default is None,
            which means that it will output all of the data.

        Returns
        -------
        None.

        '''
        if trial is None: # Save everything
            # creates the data to be worked with and arranges it vertically
            dat = np.hstack((np.arange(1,self.numTrials+1).reshape(-1,1), 
                             self.trials, 
                             np.reshape(np.array(self.actions), (-1,1)))
                            )
            np.savetxt(saveLoc, dat, delimiter=',', 
                       fmt=self._out_frmt, header=self._out_header
                       )  
        else: # Save a single trial
            # creates the data to be worked with and arranges it vertically
            dat = np.hstack(([trial], 
                             self.trials[trial-1], 
                             [self.actions[trial-1]])
                            ).reshape(-1,len(self._out_frmt))
            
            # checks to see if the file exists
            if isfile(saveLoc):
                # loads the old data, adds the new data, and then writes
                old = np.loadtxt(saveLoc, delimiter=',',dtype='str')
                dat = np.vstack((old, dat))
                np.savetxt(saveLoc, dat, delimiter=',', 
                       fmt=self._out_frmt, header=self._out_header
                       )              
            else:
                np.savetxt(saveLoc, dat, delimiter=',', 
                       fmt=self._out_frmt, header=self._out_header
                       )    
                
                
    def description(self):
        '''
        Prints a description of this task.

        Returns
        -------
        None.

        '''
        print("----------------------------------------")
        print("                  DPX")
        print("                 v ", self.__version__)
        print("----------------------------------------")
        print("Time Step Size (dt):", self.dt)
        print("Number of Trials:", self.numTrials)
        print("  AX Trials:", self.numAX)
        print("  AY Trials:", self.numAY)
        print("  BX Trials:", self.numBX)
        print("  BY Trials:", self.numBY)
        print("Intervals:")
        print("  Pre Cue:",self.preCueDur)
        print("  Cue:",self.cueDur)
        print("  ISI:",self.ISI)
        print("  Probe:",self.probeDur)
        print("  ITI (Post Probe):",self.ITI)
        print("Response Time:", self.actTime)
        
        print("Cue Map:")
        for param, val in self.cueMap.items(): 
            print("  ",param, "position:", round(val,3))
        print("----------------------------------------")
        print("")
        
class DPXwDistractor(DPX):
    """
    A task that is used to assess context integration.
    
    Attributes
    ----------
    cueMap : dict
        How cues are mapped to the agent.
    dt : float
        How quickly time changes.
    trials : int
        The number of trials
    aProp : float
        The proportion of trials with an 'A' cue.
    axProp : float
        The proportion of trials with an 'A' cue paired with an 'X' probe.
    bxProp : float
        The proportion of trials with a 'B' cue paired with an 'X' probe.
    preCueDur : float
        The time before the cue starts during each trial.
    cueDur : float
        The duration of the cue.
    ISI : float
        The duration between the cue and probe.
    probeDur : float
        The duration of the probe.
    ITI : float
        The time after the probe until the end of the trial.
    actTime : float
        When the agent's action is probed.
    rewardV : float
        The value of the reward sent to the agent.
    failV : float
        The value of a failure to the agent.
    trials : list
        The trial types.
    numAX : int
        The number of 'AX' trials.        
    numBX : int
        The number of 'BX' trials.
    numAY : int
        The number of 'AY' trials.
    numBY : int
        The number of 'BY' trials.
            
    Methods
    -------
    reset()    
        Resets the DPX task so that it is ready for another agent.
    set_trials(numTrials, aProp, axProp, bxProp)
        Prepares the DPX by allocating the trials to various cue and probe
        combinations in proportion to the number trials associated with
        each.
    set_cues(aCue, *bCues)
        Sets the cues and recreates the trials based on them.
    set_probes(xProbe, *yProbes)
        Sets the probes and recreates the trials based on them.
    current_cp()
        Returns the current trial's cue and probe.
    next_trial(agent)
        Conducts the next DPX trial. Returns -1 when all trials have been
        conducted.
    run_trial(agent, cue, probe, target)
        Runs a trial of the DPX using the specified cue and probe.
    output_data(save_loc, trial=None)
        Output the behavioral data. Can either send the entire data or one trial.
    description()
        Prints to screen a description of the task.
    """
            
    __version__ = '1.0'
    
    # Output format
    _out_frmt = ['%.8s', '%.8s', '%.8s', '%.8s', '%.8s', '%.8s']
    _out_header = 'trial,cue,probe,distractor,correct,action'
    
    # Arguments that can be used by an argument parser.
    _kwargs_list = {
        'dt': {'type': float, 'help': 'Size of the time steps.'},
        'trials': {'type': int, 'help': 'The number of trials to conduct.'},
        'aProp': {'type': float, 
                  'help': 'The proportion of trials that use the A cue.'},
        'axProp': {'type': float, 
                  'help': 'The proportion of A cue trials that have X probes.'},
        'bxProp': {'type': float, 
                  'help': 'The proportion of B cue trials that have X probes.'}, 
        'dPre': {'type': int, 
                  'help': 'Duration of time prior to the cue. (in ms)'}, 
        'dCue': {'type': int, 
                  'help': 'Duration of cue presentation. (in ms)'}, 
        'dISI': {'type': int, 
                  'help': 'Duration of beginning of interstimulus interval. (in ms)'}, 
        'dDist': {'type': int, 
                  'help': 'Duration of the distractor. (in ms)'}, 
        'dISI2': {'type': int, 
                  'help': 'Duration of end of interstimulus interval. (in ms)'}, 
        'dProbe': {'type': int, 
                  'help': 'Duration of probe presentation. (in ms)'}, 
        'dITI': {'type': int, 
                  'help': 'Duration of the intertrial interval. (in ms)'}, 
        'actTime': {'type': int, 
                  'help': 'When the agent can start to act. (in ms)'},  
        'rewardV': {'type': float, 'help': 'Value of the reward'},  
        'failV': {'type': float, 'help': 'Value of a failure'},
            }
    
    def __init__(self, cueMap, dDist=250, dISI2=1875, **kwargs):
        # Set up the cues and probes
        self._distractors = np.array(['0','1'])
        
        kwargs.setdefault('dISI', 1875)
        super().__init__(cueMap, **kwargs)
        
        self.distDur = dDist
        self.ISI2 = dISI2
        
    def set_trials(self, numTrials, aProp, axProp, bxProp):
        '''
        Prepares the DPX by allocating the trials to various cue and probe
        combinations in proportion to the number trials associated with
        each.

        Parameters
        ----------
        numTrials : integer
            The number of trials that will be run.
        aProp : float
            The proportion of trials that will have the 'A' cue.
        axProp : flaot
            The proportion of 'A' cue trials that will have an 'X' probe.
        bxProp : float
            The proportion of 'B' cue trials that will have an 'X' probe.

        Returns
        -------
        None.

        '''
        
        # Ensure that the data passed in is of the correct format.
        if not isinstance(numTrials, int):
            raise ValueError('numTrials is not an integer.')
        if aProp < 0 or aProp > 1:
            raise ValueError('aProp is not between zero and one.')
        if axProp < 0 or axProp > 1:
            raise ValueError('axProp is not between zero and one.')
        if bxProp < 0 or bxProp > 1:
            raise ValueError('bxProp is not between zero and one.')
        
        # Set up the trial proportions
        self.numTrials = numTrials
        self.aProp, self.axProp, self.bxProp = aProp, axProp, bxProp
        self.bProp = 1 - self.aProp 
        self.ayProp = 1 - self.axProp
        self.byProp = 1 - self.bxProp
        
        # Trial Parameters
        self.numAX = round(numTrials * self.aProp * self.axProp)
        self.numAY = round(numTrials * self.aProp * self.ayProp)
        self.numBX = round(numTrials * self.bProp * self.bxProp)
        self.numBY = (numTrials 
                      - (self.numAX + self.numAY + self.numBX))

        # Set up the trials and cue/probe info
        self.cues = np.append(self._aCue, self._bCues)
        self.probes = np.append(self._xProbe, self._yProbes)
        
        # Sets up the trial types
        self._AX = np.array(np.meshgrid(self._aCue, 
                                        self._xProbe,
                                        self._distractors,
                                        self._targResp)
                            ).T.reshape(-1,4)
        self._AY = np.array(np.meshgrid(self._aCue, 
                                        self._yProbes,
                                        self._distractors,
                                        self._ntargResp)
                            ).T.reshape(-1,4)
        self._BX = np.array(np.meshgrid(self._bCues, 
                                        self._xProbe,                                        
                                        self._distractors,
                                        self._ntargResp)
                            ).T.reshape(-1,4)
        self._BY  = np.array(np.meshgrid(self._bCues, 
                                         self._yProbes,
                                         self._distractors,
                                         self._ntargResp)
                             ).T.reshape(-1,4)
        
        # Creates the trials array and shuffles
        self.trials = np.zeros([0,4], dtype = '<U2')

        # Adds the AX trials        
        for i in range(int(self.numAX/self._AX.shape[0])): 
            self.trials = np.vstack((self.trials,self._AX))
        if self.numAX % self._AX.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numAX 
                                                       % self._AX.shape[0]), 
                                                      self._AX)
                                    ))
        
        # Adds the AY trials
        for i in range(int(self.numAY/self._AY.shape[0])): 
            self.trials = np.vstack((self.trials,self._AY))
        if self.numAY % self._AY.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numAY
                                                       % self._AY.shape[0]), 
                                                      self._AY)
                                    ))
        # Adds the BX trials    
        for i in range(int(self.numBX/self._BX.shape[0])):  
            self.trials = np.vstack((self.trials,self._BX))
        if self.numBX % self._BX.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numBX
                                                       % self._BX.shape[0]), 
                                                      self._BX)
                                    ))
        # Adds the BY trials
        for i in range(int(self.numBY/self._BY.shape[0])):  
            self.trials = np.vstack((self.trials,self._BY))
        if self.numBY % self._BY.shape[0] > 0:
            self.trials = np.vstack((self.trials,
                                     DPX._samp_wo_rep((self.numBY
                                                       % self._BY.shape[0]), 
                                                      self._BY)
                                    ))
        
        # Shuffles the trials and sets the current trial number
        rand.shuffle(self.trials)
        self._currentTrial = -1
        
    def set_distractors(self, *distractors):
        '''
        Sets the distractors and  recreates the trials.

        Parameters
        ----------
        *distractors : str
            Strings that represent the distractors. 

        Returns
        -------
        None.

        '''        
        self._distractors = np.array(distractors)
        self.set_trials(self.numTrials, self.aProp, self.axProp, self.bxProp)   
 
    def run_trial(self, agent, cue, probe, distractor, target):
        '''
        Runs a trial of the DPX using the specified cue and probe.

        Parameters
        ----------
        agent : object
            The agent that will be interacting with the DPX.
        cue : str
            The cue that will be presented during the trial. Must be
            in the cueMap.
        probe : str
            The probe that will be presented during the trial. Must be
            in the cueMap.
        distractor : str
            The distractor that will be presented during the trial. Must be
            in the cueMap.
        target : str
            The desired response for this cue-probe pair. 

        Returns
        -------
        action : str
            The agent's action during the trial.
        '''

        # Reset the trial information
        _tDur = 0
        _acted = False
        action = None
        
        # Execute the trial
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.preCueDur, 
                                              _tDur, _acted)
        agent.present_cue(self.cueMap.get(cue))
        action, _acted, _tDur = self._process(agent, self.cueDur, 
                                              _tDur, _acted)
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.ISI, 
                                              _tDur, _acted)
        agent.present_cue(self.cueMap.get(distractor))
        action, _acted, _tDur = self._process(agent, self.distDur, 
                                              _tDur, _acted)
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.ISI2, 
                                              _tDur, _acted)
        agent.present_cue(self.cueMap.get(probe))
        action, _acted, _tDur = self._process(agent, self.probeDur, 
                                              _tDur, _acted) 
        agent.hide_cue()
        action, _acted, _tDur = self._process(agent, self.ITI, 
                                              _tDur, _acted)               
        
        return action
                
    def description(self):
        '''
        Prints a description of this task.

        Returns
        -------
        None.

        '''
        print("----------------------------------------")
        print("           DPX w/ Distractors")
        print("                 v ", self.__version__)
        print("----------------------------------------")
        print("Time Step Size (dt):", self.dt)
        print("Number of Trials:", self.numTrials)
        print("  AX Trials:", self.numAX)
        print("  AY Trials:", self.numAY)
        print("  BX Trials:", self.numBX)
        print("  BY Trials:", self.numBY)
        print("Intervals:")
        print("  Pre Cue:",self.preCueDur)
        print("  Cue:",self.cueDur)
        print("  ISI_1:",self.ISI)
        print("  Distractor:",self.distDur)
        print("  ISI_2:",self.ISI2)
        print("  Probe:",self.probeDur)
        print("  ITI (Post Probe):",self.ITI)
        print("Response Time:", self.actTime)
        
        print("Cue Map:")
        for param, val in self.cueMap.items(): 
            print("  ",param, "position:", round(val,3))
        print("----------------------------------------")
        print("")