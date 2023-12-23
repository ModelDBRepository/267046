# -*- coding: utf-8 -*-
"""
30 Aug 19 - The purpose of this code is to test a single pyramidal neuron to
            see if it is spiking properly.
"""

import time

from matplotlib import pyplot as plt 
import numpy as np

from agent import LIF_Pop, kinetics

# Creates the pyramidal neu6ron
pyr = LIF_Pop.pyrPop(1, record_i=True, vTh=-50, vReset=-60)
pyr.AMPA.g = 1.5
pyr.NMDA.g = 0.3
pyr.GABA.g = 1.0

#pyr.AMPA.kinetic = currents.FirstOrdKinExp(1, 1, 2, track = True)
pyr.AMPA.kinetic = kinetics.secondOrdKin(1, 1, 1, 0.05, 2, track = True)
#pyr.noise.g = 0.001
weight = np.array([0.05]).reshape(-1,1)

# Run parameters
dt = 0.01           # in ms
duration = 500      # in ms
steps = int(duration/dt)

# Timings of afferent signals
receiveInterval = 1000/20 # the denominator is the rate (in Hz)
steady = True
nextTime = receiveInterval
startV = -55
pyr._vM = np.ones(pyr.size) * startV
clamp = False        # whether the voltage should be 'clamped' to a specific value
clampV = np.ones(pyr.size) * -55        # in mV

# Timings of afferent signals
GABAreceiveInterval = 1000/20 # the denominator is the rate (in Hz)
GABAstart = 250 # in ms
nextGABATime = GABAstart

# Tracks the membrane voltage and currents over time x-Axis range
TimePoints = []
spikeTimes = []
membraneV= []
spike_membraneV = -40

#Sets the voltage at the start of the experiment
if clamp: pyr._vM = np.copy(clampV)

print("Start Run")
start = time.time()

for i in range(steps):
    t = i * dt
    
    pyr.predictSpikes(dt, t)
    
    #Sends signal at times
    if (steady == True) and (t >= nextTime):
        nextTime = dt - (t - nextTime)
        pyr.receivePyrSpikes(dt, 1, np.array(0).reshape(-1,1), weight)
        nextTime = t + receiveInterval  
    else:
        pyr.receivePyrSpikes(dt, 0, np.array(0), weight)
        
    #Sends signal at times
    if (steady == True) and (t >= nextGABATime) and (t >= GABAstart):
        nextGABATime = dt - (t - nextGABATime)
        pyr.receiveIntSpikes(dt, 1, np.array(0).reshape(-1,1), weight)
        nextGABATime = t + GABAreceiveInterval  
    else:
        pyr.receivePyrSpikes(dt, 0, np.array(0), weight)
        
    pyr.processTimeStep(dt, t)
    if clamp: pyr._vM = np.copy(clampV)
    
    #Collect data
    TimePoints.append(t)
    if pyr.spikes()[0] == True: 
        membraneV.append(spike_membraneV)
        spikeTimes.append(t)
    else: 
        membraneV.append(pyr.VMs()[0])    
    
print("Duration:", time.time() - start)

# Display Parameters
xLabel = r'Time (in ms)'
yLabel = 'Membrane Potential'

# Plot the functions and add a reference line
plt.figure(figsize=(2.5, 0.5))
plt.plot(TimePoints, membraneV, color='black', linewidth = 1)
plt.plot(spikeTimes, np.ones(len(spikeTimes)) * spike_membraneV, ls='', 
         marker='D', ms=3,color='red', zorder=10, clip_on=False)
plt.ylim(-61, spike_membraneV)
plt.yticks(np.arange(-60, -39, 20))
plt.xticks([0, 250, 500])

# Label everything
plt.xlabel(xLabel) 
plt.ylabel(yLabel) 
  
# Show the plot 
#plt.savefig('./Figures/EI_CP_Figures/spiking.pdf', transparent=True)
plt.show()


# Display Parameters
xLabel = r'Time (in ms)'
yLabel = 'Current'

# Plot the functions and add a reference line
#if not hideAffCur: plt.plot(TimePoints, afferentCurrent, color='black', linewidth = 1)
#if not hideAffCur: plt.plot(TimePoints, deltaV, color='black', linestyle='dashed', linewidth = 2)
plt.figure(figsize=(2.5, 1.5))
plt.plot(TimePoints, pyr.AMPA.curr_record, color='black', linewidth = 1)
plt.plot(TimePoints, pyr.NMDA.curr_record, color='blue', linewidth = 1)
plt.plot(TimePoints, pyr.GABA.curr_record, color='red', linewidth = 1)
plt.plot(TimePoints, pyr.leak.curr_record, color='orange', linewidth = 1)
plt.ylim(1.1, -1.1)
plt.xticks([0, 250, 500])

# Label everything
plt.xlabel(xLabel) 
plt.ylabel(yLabel) 
  
# Show the plot
#plt.savefig('./Figures/EI_CP_Figures/current.pdf', transparent=True)
plt.show()

'''
# Display Parameters
xLabel = r'Time (in ms)'
yLabel = 's Kinetic Value'
plt.figure()
plt.plot(TimePoints, pyr.AMPA.kinetic.trackedS, color='green')
plt.plot(TimePoints, pyr.NMDA.kinetic.trackedS, color='blue')
plt.plot(TimePoints, pyr.GABA.kinetic.trackedS, color='red')
plt.xlim(receiveInterval-1, receiveInterval*2 - 1)
plt.ylim(0, 0.1)
plt.show()


# Display Parameters
xLabel = r'Time (in ms)'
yLabel = 's Kinetic Value'
title = 'AMPA Kinetics'

# Plot the functions and add a reference line
plt.figure()
#plt.plot(TimePoints, pyr.AMPA.kinetic.trackedX, color='green', linewidth = 1)
plt.plot(TimePoints, pyr.AMPA.kinetic.trackedS, color='blue', linewidth = 1)
plt.xlim(0, 200)

# Label everything
plt.xlabel(xLabel) 
plt.ylabel(yLabel) 
plt.title(title)
  
# Show the plot 
plt.show()

# Display Parametersz
xLabel = r'Time (in ms)'
yLabel = 's Kinetic Value'
title = 'NMDA Kinetics'

# Plot the functions and add a reference line
plt.figure()
plt.plot(TimePoints, pyr.NMDA.kinetic.trackedX, color='green', linewidth = 1)
plt.plot(TimePoints, pyr.NMDA.kinetic.trackedS, color='blue', linewidth = 1)
plt.xlim(0,200)

# Label everything
plt.xlabel(xLabel) 
plt.ylabel(yLabel) 
plt.title(title)
  
# Show the plot 
plt.show()
'''