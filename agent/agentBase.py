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
              
22 Apr 2021 - Separated AgentBase from more specific agents.
            
"""

__version__ = "1.0"

class agentBase(object):
    """
    Base class for agents. Provides some basic utilities and defines
    common methods (while not implementing them).
    
    Attributes
    ----------
    act_map : dict
        The mapping between the agent's action and the task. Provides
        a response that matches the task
    
    Methods
    -------
    __init__(act_map)
        Initialize the agent_base class.
        
    Class Methods
    -------------
    fill_parser(parser, required=False)
        Adds the arguments that go with this class to the argument parser.
    pop_kwargs(kwargs)
        Pops the kwargs that go with this class from those passed to the
        program.
        
    Place Holder Methods
    --------------------
    description() - Prints a description of the agent.
    pull_data() - Extracts data from the agent.
    act() - Checks to see whether the agent is prepared to act.
    present_cue() - Receive information from the world.
    hide_cue() - Remove information about the world.
    reward() - Reinforce the agent.
    state_reset() - Resets the agent for the next trial.
    full_reset() - Resets the agent for the next experiment.
    
    """
    _init_kwargs = []
    
    def __init__(self, act_map={}):
        '''
        Initialize the agent_base class.

        Parameters
        ----------
        act_map : dict
            The mapping between the agent's action and the task. Provides
            a response that matches the task

        '''
        self.act_map = act_map
    
    def act(self): 
        '''
        Checks to see whether the agent is prepared to act.

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call act with the base class.')
       
    def present_cue(self): 
        '''
        Receive information from the world.
            
        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call present_cue with the base class.')
    
    def hide_cue(self): 
        '''
        Remove information about the world.
            
        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call hide_cue with the base class.')

    def reward(self):
        '''
        Receive a reward from the world.

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.
            
        '''
        
        raise NotImplementedError('Trying to call reward with the base class.')
            
    def pull_data(self): 
        '''
        Extracts data from the agent.

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call pull_data with the base class.')

    def state_reset(self):
        '''
        Resets the agent for the next trial.

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call state_reset with the base class.')
    
    def full_reset(self): 
        '''
        Resets the agent for the next experiment

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.

        '''
        raise NotImplementedError('Trying to call full_reset with the base class.')
    
    def description(self):
        '''
        Prints a description of the agent to the console.

        Raises
        ------
        NotImplementedError
            This is the base class and this method is not implemented.
            
        '''
        raise NotImplementedError('Trying to call description with the base class.')
       
    @classmethod
    def fill_parser(cls, parser, required=False):        
        '''
        Adds the arguments that go with this class to the argument parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser
        required : bool, optional
            Whether the argument is required. The default is False.

        '''
        _pref = '--'
        if required: _pref = ''
        
        for keys, items in cls._init_kwargs.items():
            parser.add_argument(_pref + keys,
                                type=items['type'],
                                help=items['help']
                                )
        
    @classmethod
    def pop_kwargs(cls, kwargs):
        '''
        Pops the kwargs that go with this class from those passed to the
        program.

        Parameters
        ----------
        kwargs : dict
            Key word arguments that are to be parsed

        Returns
        -------
        pop_kwargs : dict
            Removed kwargs that belong to this class.

        '''
        pop_kwargs = {}
        delete = []

        # check to see if any of the kwargs belong to this class
        for keys, items in cls._init_kwargs.items():
            if keys in kwargs:
                pop_kwargs.update({keys:kwargs.pop(keys)})
                if pop_kwargs[keys] is None:
                    delete.append(keys)
        
        # strip out all of the empty kwargs
        for k in delete:
            del pop_kwargs[k]
        
        return pop_kwargs