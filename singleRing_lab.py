# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:19:19 2019

@author: ocalvin
"""

import argparse
import time
import os

import numpy as np

from agent import CueProbe_Agent
from world import DPX
from DPXAnalysis2021 import cp_raster_plot

# -------------- Parameters ----------------

# Create the mappings between the agent and world
cueMap = {'A': np.pi/2, 
          'B': np.pi/2, 
          'X': 3/2 * np.pi, 
          'Y': 3/2 * np.pi
          }
actMap = {0: 'O', 1: 'L', 2: 'R'}

# Output Control
shwTrials = True   # Assumes that you only want to run a single experiment
collectData = False

# Unstable Firing
min_excess_rate = 20 # average hertz
max_excess_count = 5 # early quit point
excess_count = 0

# ------------- Parse Program Arguments --------------------

parser = argparse.ArgumentParser(
        prog='SingleRing_lab.py', 
        description='Runs a single-ring agent on a series of AX DPX task trials.', 
        usage='%(prog)s outfolder [options]'
        )

name = 'outfolder'
if not collectData: name = '--' + name
parser.add_argument(
     name, 
     type=str,
     help='Folder that recorded data will be stored in.'
     )

# Add the parser arguments for the world
dpxGroup = parser.add_argument_group('DPX Task')
DPX.fill_parser(dpxGroup)

# Add the parser arguments for the agent
agentGroup = parser.add_argument_group('Single-Ring Agent')
CueProbe_Agent.fill_parser(agentGroup)

args = vars(parser.parse_args())

# ------------------ Incorporate arguments ------------------------

# output location for this trial
outputFolder = args['outfolder']

# Set the World defaults for this experiment
popped = DPX.pop_kwargs(args)
popped.setdefault('trials', 100)
popped.setdefault('aProp', 1.0)
popped.setdefault('axProp', 1.0)
popped.setdefault('dCue', 500)
popped.setdefault('dISI', 2000)
popped.setdefault('dITI', 500)

# Create the world
dpxTask = DPX(cueMap, **popped)

# Set the Agent defaults for this experiment
popped = CueProbe_Agent.pop_kwargs(args)

# If running a local test use these parameters
if shwTrials:    
    popped.setdefault('p_AMPAg', 0.45)
    popped.setdefault('p_NMDAg', 0.27375)
    popped.setdefault('p_GABAg', 1.25)
    popped.setdefault('i_AMPAg', 0.495)
    popped.setdefault('i_NMDAg', 0.225)
    popped.setdefault('i_GABAg', 1.00)
    
# Create the agent
rAgent = CueProbe_Agent(act_map=actMap, **popped)

# ------------------ Print Details -------------------------
dpxTask.description()
rAgent.description()

# ------------------ Start the Task ------------------------

# Creates the directory if it doesn't exist yet
if collectData and not os.path.exists(outputFolder): os.makedirs(outputFolder)

# Set the dpxParameters
tskDur = (dpxTask.preCueDur + 
          dpxTask.cueDur + 
          dpxTask.ISI + 
          dpxTask.probeDur +
          dpxTask.ITI
          )

# Run the trials
for t in range(dpxTask.numTrials):       
    startTime = time.time()
    
    # Reset the state of the agent before starting the trial
    rAgent.state_reset()
    dpxTask.next_trial(rAgent)
        
    # If the data is being collected
    if collectData:
        p_rec, i_rec = rAgent.pull_data()
        
        # collects the spiking information
        p_rec.save(outputFolder + 'pyr.csv', t+1)
        i_rec.save(outputFolder + 'int.csv', t+1)
        
        # Calculate the average rate of firing for each neuron
        rate = rAgent._trackPyr.raster.size / rAgent.percPyr.size 
        rate /= (tskDur / 1000)

    # Determine the duration of the trial and update kick out an update to 
    #   let the user know the status of the program.
    print("Trial ", t+1, " duration: ", 
          round(time.time() - startTime), " seconds"
          )
        
    if shwTrials:
        # Raster plot
        p_rec, i_rec = rAgent.pull_data()
        cp_raster_plot(p_rec)
        
        break  # only run one trial
        
    # Kickout early if the network is continuously firing.
    if rate > min_excess_rate:
        excess_count += 1
        if excess_count >= max_excess_count:
            print("Early kickout due to excess firing.")
            break