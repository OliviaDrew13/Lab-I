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
from scipy import signal

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams["axes.formatter.limits"] = -2, 2

def plotting(x, y, xyLabels, labels, types, labelsOn = True, xExtents = None, yExtents = None, grid = True, title = None, sharex = False, aspect = None):
    
    
    title = None
    fig, ax = plt.subplots()
        
    if np.shape(labels) == ():
        print("You have not provided enough labels, duplicating...")
        labelVal = labels
        labels = []
        for i in range(len(y)):
            labels.append(labelVal)
            
    if np.shape(types) == ():
        typeVal = types
        types = []
        for i in range(len(y)):
            types.append(typeVal)
            
    if sharex == True:
        xs=[x]
        for i in range(len(y)):
            xs.append(x)
        x = xs
            
    # looping through the data to be plot and checking plottype
    for i in range(len(y)):
        
        if types[i] == 'p':
            ax.plot(x[i], y[i], label = labels[i])
            
        elif types[i] == 's':
            ax.scatter(x[i], y[i], s=5, label = labels[i])
            
        elif types[i] == 'l':
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
    
    if aspect != None:
        ax.set_aspect("equal")
    
    # Setting the plotting extents if they are specified
    if xExtents != None:
        ax.set_xlim(xExtents[0], xExtents[1])
    if yExtents != None:
        ax.set_ylim(yExtents[0], yExtents[1])
    
    # Applying legend if true
    if labelsOn == True:
        ax.legend()
    
    plt.show()
    
def subplotting(x, y, xylabels, labels=None, xExtents=None, yExtents=None, labelsOn=True, grid = True, figsize = (6,5), title=None):
    
    fig, axs = plt.subplots(len(x), figsize = figsize)
    for i in range(len(x)):
        lines = []
        for n in range(len(x[i])):
            
            line, = axs[i].plot(x[i][n], y[i][n])
            lines.append(line)
        
        axs[i].set_xlabel(xylabels[0])
        axs[i].set_ylabel(xylabels[1][i])
        
        if grid == True:
            axs[i].grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
            axs[i].grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
            axs[i].minorticks_on()
        
# =============================================================================
#         # Setting the plotting extents if they are specified
#         if xExtents != None:
#             axs[i].set_xlim(xExtents[i][0], xExtents[i][1])
#         if yExtents != None:
#             axs[i].set_ylim(yExtents[i][0], yExtents[i][1])
# =============================================================================
        
        # Applying legend if true
        if labelsOn == True:
            axs[i].legend(lines, labels[i])
            print(labels[i])
    
    axs[0].set_title(" \n ")
    
    fig.tight_layout()    

def rotation(time, aX, aY, aZ, gX, gY, gZ):
    
    #flipping signs
    gX, gY, gZ = -gX, -gY, -gZ
    
    #finding the respective rotational angles in radians
    rX = np.deg2rad(scI.cumulative_trapezoid(gX, None, dt, initial=0))
    rY = np.deg2rad(scI.cumulative_trapezoid(gY, None, dt, initial=0))
    rZ = np.deg2rad(scI.cumulative_trapezoid(gZ, None, dt, initial=0))
    # rX = np.deg2rad(gX)
    # rY = np.deg2rad(gY)
    # rZ = np.deg2rad(gZ)
    print(len(rX))
    #setting up acceleration vectors
    rAX = []
    rAY = []
    rAZ = []
    
    gravity = 9.81
    
    for i in range(len(aX)):
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
    
    # rAX = np.array(rAX)+gravity
    # rAY = np.array(rAY)-gravity
    # rAZ = np.array(rAZ)-gravity
    
    # plotting([time.iloc[1:],time.iloc[1:],time.iloc[1:]], [rAX,rAY,rAZ], ["Time (s)", "Acceleration (m/s$^2$)"], ['$a_y$','$a_y$','$a_z$'], ["pl", "pl", "pl"])
    print(len(rAX))
    return(time,rAX,rAY,rAZ)

def integrationTrapezoid(time,X,Y,Z,dt):
    # Integrating the input data
    # XI = scI.cumulative_trapezoid(X,None,dt)
    # YI = scI.cumulative_trapezoid(Y,None,dt)
    # ZI = scI.cumulative_trapezoid(Z,None,dt)
    
    XI = scI.cumulative_trapezoid(X,time, initial=0)
    YI = scI.cumulative_trapezoid(Y,time, initial=0)
    ZI = scI.cumulative_trapezoid(Z,time, initial=0)
    
    return(time, XI, YI, ZI)

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
        
    return(time, np.array(vx[1:]), np.array(vy[1:]), np.array(vz[1:]))

def savgolFiltering(ax,ay,az):
    
    #filter paramters: window is number of coefficients, order is the polynomial order
    window = 80
    Order = 9

    #apply filter
    fAX = savgol_filter(ax, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAY = savgol_filter(ay, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fAZ = savgol_filter(az, window_length = int(window), polyorder = int(Order), mode = 'interp')

    return(fAX,fAY,fAZ)

def correcting(data, mean, std):
    dataCorrected = data - mean
    dataLM = data - (mean - std)
    dataUM = data - (mean + std)
    return dataCorrected, dataLM, dataUM

def butterLow(time, x, y, z, lowF):
    rate = len(time)/time.max()
    lowPass = signal.butter(5, [lowF], "Lp", fs = rate, output = "sos")
    filteredSigx = signal.sosfilt(lowPass, x)
    filteredSigy = signal.sosfilt(lowPass, y)
    filteredSigz = signal.sosfilt(lowPass, z)
    
    return filteredSigx, filteredSigy, filteredSigz

def dataCalibration(time, ax, ay, az, gx, gy, gz):
    
    calibIndices = (time<=2)
    
    variables = [ax[calibIndices], ay[calibIndices], az[calibIndices], gx[calibIndices], gy[calibIndices], gz[calibIndices]]
    corrections = []
    
    for variable in variables:
        
        mean = variable.mean()
        std = variable.std()
        corrections.append([mean, std])
        
    return np.array(corrections)
    
filePaths = ["DataDay1/rotX1_25_02_2025_Data.txt"]
# filePaths = ["DataDay2SRFix/positiveX1_04_03_2025_Data.txt", "DataDay2SRFix/negativeX1_04_03_2025_Data.txt", "DataDay2SRFix/positiveY1_04_03_2025_Data.txt", "DataDay2SRFix/negativeY1_04_03_2025_Data.txt", "DataDay2SRFix/positiveZ1_04_03_2025_Data.txt", "DataDay2SRFix/negativeZ1_04_03_2025_Data.txt"]

# Day2
corrections = [[np.float64(-0.013297079734907045), np.float64(0.004162259512190329)], [np.float64(0.012824755867845122), np.float64(0.003399006261014335)], [np.float64(-0.03623640496757119), np.float64(0.005982169092832332)], [np.float64(-0.011047987126242952), np.float64(0.00996556376118626)], [np.float64(-0.009217885793621493), np.float64(0.014695401851523902)], [np.float64(-0.059751018786683215), np.float64(0.010929141064857853)]]

upperMeans = [[], [], [], [], [], []]
lowerMeans = [[], [], [], [], [], []]

for file in filePaths:
    
    # Reading the datafile
    dataF = pd.read_csv(file, sep='\\s+')
    
    # Reading raw data
    t, ax, ay, az, gx, gy, gz = np.array(dataF["Time"]), dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    dataArrays = [ax, ay, az, gx, gy, gz]
    
    dt = 0.001
    
    print(f"sample rate = {len(ax)/t.max()}")
    
    corrections = dataCalibration(t, ax, ay, az, gx, gy, gz)
    
    # Applying measured correciton factors
    for i in range(len(dataArrays)):
        dataArrays[i], lowerMeans[i], upperMeans[i] = correcting(dataArrays[i], corrections[i][0], corrections[i][1])
    
    # Reassigning the acceleration and angular velocity arrays
    cAX, cAY, cAZ, cGX, cGY, cGZ = dataArrays
    
    # Filtering the data with Savitzky-Golay
    fAX, fAY, fAZ = savgolFiltering(cAX, cAY, cAZ)
    fGX, fGY, fGZ = savgolFiltering(cGX, cGY, cGZ)
    
    # Filtering the data with Butterworth lowpass
    freq = 20
    fAX, fAY, fAZ = butterLow(t, cAX, cAY, cAZ, freq)
    fGX, fGY, fGZ = butterLow(t, cGX, cGY, cGZ, freq)
    
    # Integrating the filtered acceleration data
    vTime, vx, vy, vz = integrationTrapezoid(t, fAX, fAY, fAZ, dt)
    posTime, x,y,z = integrationTrapezoid(vTime, vx, vy, vz, dt)
    
    # Rotating the data to the lab reference frame
    raTime, rAX, rAY, rAZ = rotation(t, fAX, fAY, fAZ, fGZ, fGY, fGZ)
    
    # Integrating using cumulative trapezoid
    rvTime, rVX, rVY, rVZ = integrationTrapezoid(raTime, rAX, rAY, rAZ, dt)
    rposTime, rX,rY,rZ = integrationTrapezoid(rvTime, rVX, rVY, rVZ, dt)
    
# =============================================================================
#     # Integrating using euler's method
#     rvTime, rVX, rVY, rVZ = euler(raTime, rAX, rAY, rAZ, dt)
#     rposTime, rX,rY,rZ = euler(rvTime, rVX, rVY, rVZ, dt)
# =============================================================================
     
    # Index      = [0     , 1  , 2  , 3  , 4     , 5  , 6  , 7  , 8       , 9 , 10, 11]
    rotatedData  = [raTime, rAX, rAY, rAZ, rvTime, rVX, rVY, rVZ, rposTime, rX, rY, rZ]
    filteredData = [t     , fAX, fAY, fAZ, vTime , vx , vy , vz , posTime , x , y , z ]
    
    # Plotting the data
    
    # 2D rotated vs unrotated positions
    plotting([x, rX], [y, rY], ["x (m)", "y(m)"], ["Raw", "Rotated"], "p", aspect = 1)
    # print(rY.max())
    # Rotated positions vs time
    plotting(rposTime, rotatedData[9:12], ["Time (s)", "Position (m)"], ["$x_R$", "$y_R$", "$z_R$"], "p", sharex = True, title = "Rotated Positions")
    plotting(t, [gx, gy, gz], ["Time (s)", "Angular Velocity (m)"], ["$x$", "$y$", "$z$"], "p", sharex = True, title = "Rotated Positions")
    
    # Rotated Accelerations
    plotting(raTime, rotatedData[1:4], ["Time (s)", "Acceleration (m/s$^2$)"], ["$A_Rx$", "$A_Ry$", "$A_Rz$"], "p", title= "Rotated Accelerations", sharex = True)
    
    # Filter Demonstrations
    plotting(t, [fAX, rAX], ["Time (s)", "Acceleration (m/s$^2$)"], ["Filtered", "Rotated"], "p", title = "Filter Demonstration x", sharex = True)
    plotting(t, [ay, fAY], ["Time (s)", "Acceleration (m/s$^2$)"], ["unfiltered", "filtered"], "p", title = "Filter Demonstration y", sharex = True)
    plotting(t, [az, fAZ], ["Time (s)", "Acceleration (m/s$^2$)"], ["unfiltered", "filtered"], "p", title = "Filter Demonstration z", sharex = True)
    
    
    # Plotting the consecutive integrations
    subplotting([[raTime], [rvTime], [rposTime]], [[rAX], [rVX], [rX]], ["Time (s)", ["Acceleration (m/s$^2$)", "Velocity (m/s)", "Displacement (m)"]], labelsOn = False, title = "")
    subplotting([[raTime], [rvTime], [rposTime]], [[rAY], [rVY], [rY]], ["Time (s)", ["Acceleration (m/s$^2$)", "Velocity (m/s)", "Displacement (m)"]], labelsOn = False, title = "Integration in $y$")
    subplotting([[raTime], [rvTime], [rposTime]], [[rAZ], [rVZ], [rZ]], ["Time (s)", ["Acceleration (m/s$^2$)", "Velocity (m/s)", "Displacement (m)"]], labelsOn = False, title = "Integration in $z$")
    
    # Plotting the path in 3d
    # plotting3D(rX, rY, rZ, ["$x_R$", "$y_R$", "$z_R$"])
    
    plotting(t, [ax, cAX], ["Time (s)", "Acceleration (m/s$^2$)"], ["Raw Data", "Corrected Data"], "p", sharex = True)
    subplotting([[t], [t]], [[fAY], [rAY]], ["Time (s)", ["Acceleration (m/s$^2$)", "Acceleration (m/s$^2$)"]], [["Filtered"], ["Rotated"]])
    
    print(f"The max acceleration is: {fAZ.max()}\nThe max angular velocity is: {fGZ.max()}")
