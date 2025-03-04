# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:46:05 2025

@author: brada
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scI
from scipy.signal import savgol_filter

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
        ax.set_xlim(extents[0], extents[1])
        ax.set_ylim(extents[2], extents[3])
    
    # Applying legend if true
    if labelsOn == True:
        ax.legend()
    
def savgolFiltering(ax,ay,az):
    
    #filter paramters: window is number of coefficients, order is the polynomial order
    window = 80
    Order = 9

    #apply filter
    fAX = savgol_filter(ax, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAY = savgol_filter(ay, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAZ = savgol_filter(az, window_length = int(window), polyorder = int(Order), mode = 'interp')

    return(fAX,fAY,fAZ)

AXmean = []
AYmean = []
AZmean = []
GXmean = []
GYmean = []
GZmean = []
AXstd = []
AYstd = []
AZstd = []
GXstd = []
GYstd = []
GZstd = []
grav = 9.81

for i in range(7):
    
    file = "DataDay2/stationaryZ"+str(i+1)+"_04_03_2025_Data.txt"
    
    dataF = pd.read_csv(file, sep='\\s+')
    #dataF = dataF.replace({"['inf']": np.nan})
    #dataF = dataF.dropna() 
    
    # Reading raw data
    t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    
    ax, ay, az = savgolFiltering(ax, ay, az)
    gx, gy, gz = savgolFiltering(gx, gy, gz)
    
    plotting([t,t,t], [ax,ay,az], ["Time (s)", "Acceleration"], ["ax", "ay", "az"], ["pl", "pl", "pl"])
    
    AXmean.append(ax.mean())
    AYmean.append(ay.mean())
    AZmean.append(az.mean())
    GXmean.append(gx.mean())
    GYmean.append(gy.mean())
    GZmean.append(gz.mean())
    AXstd.append(ax.std())
    AYstd.append(ay.std())
    AZstd.append(az.std())
    GXstd.append(gx.std())
    GYstd.append(gy.std())
    GZstd.append(gz.std())

mainArr = np.array([[AXmean, AXstd], [AYmean, AYstd], [AZmean, AZstd], [GXmean, GXstd], [GYmean, GYstd], [GZmean, GZstd]])

finalVals = []

for pair in mainArr:
    aveVal = pair[0].mean()
    valErr = np.sqrt(np.sum((pair[1]/len(pair[0]))**2))
    finalVals.append([aveVal, valErr])
    
finalVals[2][0] = finalVals[2][0] - grav
print("ax, ay, az, gx, gy, gz")
print(finalVals)
    