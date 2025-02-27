# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:25 2025

@author: Sam
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
    
def rotation(time, aX, aY, aZ, gX, gY, gZ):
    
    #flipping signs
    gX, gY, gZ = -gX, -gY, -gZ
    
    #finding the respective rotational angles in radians
    rX = np.deg2rad(scI.cumulative_trapezoid(gX, None, dt))
    rY = np.deg2rad(scI.cumulative_trapezoid(gY, None, dt))
    rZ = np.deg2rad(scI.cumulative_trapezoid(gZ, None, dt))
    # rX = np.deg2rad(gX)
    # rY = np.deg2rad(gY)
    # rZ = np.deg2rad(gZ)

    #setting up acceleration vectors
    rAX = []
    rAY = []
    rAZ = []
    
    gravity = 9.81
    
    for i in range(len(aX)-1):
        accVec = []
        accVec = np.array([aX[i], aY[i], aZ[i]])
        rotMatX = np.array([[1, 0, 0], [0, np.cos(rX[i]), -np.sin(rX[i])], [0, np.sin(rX[i]), np.cos(rX[i])]])
        rotMatY = np.array([[np.cos(rY[i]), 0, np.sin(rY[i])], [0, 1, 0], [-np.sin(rY[i]), 0, np.cos(rY[i])]])
        rotMatZ = np.array([[np.cos(rZ[i]), -np.sin(rZ[i]) ,0], [np.sin(rZ[i]), np.cos(rZ[i]), 0], [0, 0, 1]])
        
        # rotatedZ = np.matmul(rotMatZ, accVec)
        # rotatedY = np.matmul(rotMatY, rotatedZ)
        # rotatedX = np.matmul(rotMatX, rotatedY)
        
        rotatedX = np.matmul(rotMatX, accVec)
        rotatedY = np.matmul(rotMatY, rotatedX)
        rotatedZ = np.matmul(rotMatZ, rotatedY)
        
        rAX.append(rotatedZ[0])
        rAY.append(rotatedZ[1])
        rAZ.append(rotatedZ[2])
        
    print(np.mean(np.array(rAZ)))
    
    rAX = np.array(rAX)-gravity
    
    # plotting([time.iloc[1:],time.iloc[1:],time.iloc[1:]], [rAX,rAY,rAZ], ["Time (s)", "Acceleration (m/s$^2$)"], ['$a_y$','$a_y$','$a_z$'], ["pl", "pl", "pl"])
    
    return(time.iloc[1:],rAX,rAY,rAZ)

def integrationTrapezoid(time,X,Y,Z,dt):
    # Integrating the input data
    XI = scI.cumulative_trapezoid(X,None,dt)
    YI = scI.cumulative_trapezoid(Y,None,dt)
    ZI = scI.cumulative_trapezoid(Z,None,dt)
    
    # XI = scI.cumulative_trapezoid(X,time)
    # YI = scI.cumulative_trapezoid(Y,time)
    # ZI = scI.cumulative_trapezoid(Z,time)
    
    timeI = time[1:]
    
    return(timeI, XI, YI, ZI)

def plotting3D(x, y, z, axisLabels):
    
    # creating a 3d plot and plotting the entered data
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)), zoom=0.85)
    
    ax.plot(x,y,z)
    ax.set_xlabel(axisLabels[0])
    ax.set_ylabel(axisLabels[1])
    ax.set_zlabel(axisLabels[2])
    ax.set_yticks([y.min(),y.max()])
    
def euler(time, aX, aY, aZ, dt):
    
    
    vx=[0]
    vy=[0]
    vz=[0]
    
    for i in range(len(aX)):
        vx.append(vx[i]+(aX[i] * dt))
        vy.append(vy[i]+(aY[i] * dt))
        vz.append(vz[i]+(aZ[i] * dt))
        
    return(vx,vy,vz)

def savgolFiltering(ax,ay,az):
    
    #filter paramters: window is number of coefficients, order is the polynomial order
    window = 80
    Order = 9

    #apply filter
    fAX = savgol_filter(ax, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAY = savgol_filter(ay, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAZ = savgol_filter(az, window_length = int(window), polyorder = int(Order), mode = 'interp')

    return(fAX,fAY,fAZ)

# filePaths = ["DataDay1/xMoveTest1_25_02_2025_Data.txt"]
filePaths = ["DataDay1/gYTest1_25_02_2025_Data.txt"]

for file in filePaths:
    
    dataF = pd.read_csv(file, sep='\\s+')
    #dataF = dataF.replace({"['inf']": np.nan})
    #dataF = dataF.dropna() 
    
    # Reading raw data
    t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    
    print(np.mean(gz))
    
    dt = 0.001
    
    # Filtering the data with Savitzky-Golay
    fAX, fAY, fAZ = savgolFiltering(ax,ay,az)
    fGX, fGY, fGZ = savgolFiltering(gx,gy,gz)
    
    # Accounting for g
    # fAZ = np.array(fAZ) - 9.81
    
    # Integrating the filtered acceleration data
    vTime, vx, vy, vz = integrationTrapezoid(t, fAX, fAY, fAZ, dt)
    posTime, x,y,z = integrationTrapezoid(vTime, vx, vy, vz, dt)
    
    # Using the Euler Method
    vxEuler, vyEuler, vzEuler = euler(t,fAX,fAY,fAZ,dt)
    
    # Rotating the data to the lab reference frame
    raTime, rAX, rAY, rAZ = rotation(t, fAX, fAY, fAZ, fGZ, fGY, fGZ)
    rvTime, rVX, rVY, rVZ = integrationTrapezoid(raTime, rAX, rAY, rAZ, dt)
    rposTime, rX,rY,rZ = integrationTrapezoid(rvTime, rVX, rVY, rVZ, dt)
    
    plotting([x, rX], [y, rY], ["x", "y"], ["Unrotated", "Rotated"], ["pl", "pl"])
    plotting([posTime, posTime, posTime], [x, y, z], ["Time (s)", "Position (m)"], ["x", "y", "z"], ["pl", "pl", "pl"])
    plotting([t, t], [fAX, fAZ], ["Time (s)", "Acceleration (m/s^2)"], ["ax", "az"], ["pl", "pl", "pl"])
    plotting([rposTime, rposTime, rposTime], [rX, rY, rZ], ["Time (s)", "Position (m)"], ["$x_R$", "$y_R$", "$z_R$"], ["pl", "pl", "pl"])
    plotting([raTime, raTime, raTime], [rAX, rAY, rAZ], ["Time (s)", "Acceleration (m/s^2)"], ["$A_Rx$", "$A_Ry$", "$A_Rz$"], ["pl", "pl", "pl"])
    
    plotting3D(x,y,z,["x", "y", "z"])
    plotting3D(rX, rY, rZ, ["$x_R$", "$y_R$", "$z_R$"])
    
    print(f"The max acceleration is: {fAZ.max()}\nThe max angular velocity is: {fGZ.max()}")
