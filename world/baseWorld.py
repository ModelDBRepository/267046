# -*- coding: utf-8 -*-
"""
Purpose
-------
Holds the BaseWorld class which provides some useful utilities.

Classes
-------
BaseWorld - Base class for worlds. Provides some automatic argument 
            parsing for commmand line interfaces.

Functions
---------
None

Change log
----------
2021 Feb 12 - Reorganized files. Split off the baseWorld from the DPX classes.

"""

class BaseWorld(object):
    '''
    Base class for worlds. Mostly exists to serve as a template for how 
    these should be written and provide some automatic argument parsing.
    '''
    
    _kwargs_list = []
    
    def __init__(self, cueMap, dt=0.05, **kwargs):
        self.cueMap = cueMap
        self.dt = dt
        
    def start(self, agent): 
        '''
        Starts the task.

        Parameters
        ----------
        agent : object
            The agent that will be interacting with this world.
        '''
        raise Exception('Trying to call start with the base class')

    def reset(self):
        '''
        Reset the world so that it is ready for the next participant
        '''
        raise Exception('Trying to call reset with the base class')
    
    def output_data(self):
        '''
        Creates the data that will be saved.

        Returns
        -------
        Data in the format it will be saved.

        '''
        raise Exception('Trying to call output_data with the base class')
    
    @classmethod
    def fill_parser(cls, parser, required=False):        
        '''
        Adds the arguments that go with this class to the command line
        argument parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser
        required : bool, optional
            Whether the argument is required. The default is False.

        Returns
        -------
        None.

        '''
        _pref = '--'
        if required: _pref = ''
        
        for keys, items in cls._kwargs_list.items():
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
        for keys, items in cls._kwargs_list.items():
            if keys in kwargs:
                pop_kwargs.update({keys:kwargs.pop(keys)})
                if pop_kwargs[keys] is None:
                    delete.append(keys)
        
        # strip out all of the empty kwargs
        for k in delete:
            del pop_kwargs[k]
        
        return pop_kwargs
    
    def description(self):
        '''
        Prints a text description of the world's settings'

        Returns
        -------
        string
            A text description of the parameters that the world has been
            set with.

        '''
        raise Exception('Trying to call description with the base class')