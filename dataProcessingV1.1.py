# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:25 2025

@author: Sam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams["axes.formatter.limits"] = -2, 2

def plotting(x, y, xyLabels, labels, types, labelsOn = True, extents = None, grid = True, title = None):
    
    fig, ax = plt.subplots()
    
    # looping through the data to be plot and checking plottype
    for i in range(len(x)):
        
        if types[i] == 'pl':
            ax.plot(x[i], y[i], label = labels[i])
            
        elif types[i] == 'sc':
            ax.scatter(x[i], y[i], s=5, label = labels[i])
            
        elif types[i] == 'vl':
            ax.vlines(x[i], 0, 1, color='grey', zorder=0, linestyle = '-.', lw=0.5, label = labels[i])
    
    # Checking whether to plot a grid
    if grid == True:
        ax.grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
        ax.grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
        ax.minorticks_on()
        
    # Setting the axes below the data
    ax.set_axisbelow(True)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_title(title)
    
    # Setting the plotting extents if they are specified
    if extents != None:
        ax.set_xlim(extents[0], np.round(x[0].max() + 0.4))
        ax.set_ylim(extents[2], extents[3])
    
    # Applying legend if true
    if labelsOn == True:
        ax.legend()

filePaths = ["DataDay1/rotY1_25_02_2025_Data.txt"]

for file in filePaths:
    
    dataF = pd.read_csv(file, sep='\\s+')
    #dataF = dataF.replace({"['inf']": np.nan})
    #dataF = dataF.dropna() 
    
    t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    #plot results            
    plotting([t,t,t], [ax,ay,az], ["Time (s)", "Acceleration (m/s$^2$)"], ['$a_y$','$a_y$','$a_z$'], ["pl", "pl", "pl"])
    plotting([t,t,t], [gx,gy,gz], ["Time (s)", "Angular Velocity (deg/s)"], ['$\\omega_x$','$\\omega_y$','$\\omega_z$'], ["pl", "pl", "pl"])
    
    print(f"The max acceleration is: {az.max()}\nThe max angular velocity is: {gz.max()}")
    
