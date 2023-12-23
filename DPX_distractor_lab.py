# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:19:19 2019

@author: ocalvin
"""

import time
import os
import argparse

import numpy as np

from agent import DPX_Agent
from world import DPXwDistractor
from DPXAnalysis2021 import dpx_raster_plot

# -------------- Parameters ----------------

# Create the mappings between the agent and world
stDist = 0.20
cueMap = {'A': np.pi * (0.5 - stDist), 
          'B': np.pi * (0.5 + stDist), 
          'X': np.pi * (1.5 - stDist), 
          'Y': np.pi * (1.5 + stDist),
          '0': 0,
          '1': np.pi
          }
actMap = {0: 'O', 1: 'L', 2: 'R'}

# Output Control
shwTrials = True   # Assumes that you only want to run a single experiment
collectData = False

# ------------- Parse Program Arguments --------------------

parser = argparse.ArgumentParser(
                    prog='DPX_lab.py', 
                    description='Runs a dual-ring agent on the DPX task.', 
                    usage='%(prog)s outfolder [options]'
                    )

name = 'outfolder'
if not collectData: name = '--' + name
parser.add_argument(
     name, 
     type=str,
     help='Folder that recorded data will be stored in.'
     )

dpxGroup = parser.add_argument_group('DPX w/ Distractor Task')
DPXwDistractor.fill_parser(dpxGroup)

# Add the parser arguments for the agent
agentGroup = parser.add_argument_group('Dual-Ring SoftMax Agent')
DPX_Agent.fill_parser(agentGroup)


args = vars(parser.parse_args())

# ------------------ Incorporate arguments ------------------------

# output location for this trial
outputFolder = args['outfolder']

# If running a local test use these parameters
if shwTrials:
    args.update({'aProp': 1.00})
    args.update({'axProp': 1.00})
    args.update({'bxProp': 1.00})
    #collectData = True
    #outputFolder = './test/'
    #popped.setdefault('mpNMDAg', 1.00)    

# Set the World defaults for this experiment
popped = DPXwDistractor.pop_kwargs(args)

# Create the world
dpxTask = DPXwDistractor(cueMap, **popped)

# Set the Agent defaults for this experiment
popped = DPX_Agent.pop_kwargs(args)
    
# Create the agent
rAgent = DPX_Agent(actMap, **popped)

# Set the weights for the agent's action kinetics
rAgent.set_act_weight(cueMap.get('A') - (np.pi * stDist), 
                      cueMap.get('A') + (np.pi * stDist), 
                      0.075, 'mem', 0
                      )
rAgent.set_act_weight(cueMap.get('B') - (np.pi * stDist), 
                      cueMap.get('B') + (np.pi * stDist),
                      0.375, 'mem', 1
                      )
rAgent.set_act_weight(cueMap.get('X') - (np.pi * stDist), 
                      cueMap.get('X') + (np.pi * stDist), 
                      0.05, 'perc', 0
                      )
rAgent.set_act_weight(cueMap.get('Y') - (np.pi * stDist), 
                      cueMap.get('Y') + (np.pi * stDist), 
                      0.25, 'perc', 1
                      )

# ------------------ Print Details -------------------------

dpxTask.description()
rAgent.description()

# ------------------ Start the Task ------------------------

# Creates the directory if it doesn't exist yet
if collectData and not os.path.exists(outputFolder): 
    os.makedirs(outputFolder)

# Set the dpxParameters
tskDur = (dpxTask.preCueDur 
         + dpxTask.cueDur
         + dpxTask.ISI
         + dpxTask.distDur
         + dpxTask.ISI2
         + dpxTask.probeDur 
         + dpxTask.ITI
         )

# Run the trials
for t in range(dpxTask.numTrials):       
    startTime = time.time()
    
    # Reset the state of the agent before starting the trial
    rAgent.state_reset()
    dpxTask.next_trial(rAgent)
        
    # If the data is being collected
    if collectData:
        rec_pp, rec_pi, rec_mp, rec_mi = rAgent.pull_data()
        
        # collects the spiking information
        rec_pp.save(outputFolder + 'percPyr.npz', t+1)
        rec_pi.save(outputFolder + 'percInt.npz', t+1)
        rec_mp.save(outputFolder + 'memPyr.npz', t+1)
        rec_mi.save(outputFolder + 'memInt.npz', t+1)
         
        # save the behavioral data
        dpxTask.output_data(outputFolder + 'dpx.csv', t+1)
    
    if shwTrials:        
        print("Trial duration: ", round(time.time() - startTime), " seconds")
        rec_pp, rec_pi, rec_mp, rec_mi = rAgent.pull_data()
        
        # Plot the raster plots for the perception and memory
        dpx_raster_plot(rec_pp, dpxTask.current_cp())
        dpx_raster_plot(rec_mp, dpxTask.current_cp())    
        break

    
    # Let the user know how long the trial took
    print("Trial ", t+1, " duration: ", round(time.time() - startTime), 
          " seconds")