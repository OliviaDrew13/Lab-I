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

# =============================================================================
# Plotting
# =============================================================================

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams["axes.formatter.limits"] = -2, 2

def plotting(x, y, xyLabels, types, labels=None, xExtents = None, yExtents = None, grid = True, 
             title = None, sharex = False, thickness = 1, yScale="linear", line = "-",
             xScale = "linear", invertY = False, yTicks = None, aspect = "auto"):
    
    """
    A generaliseable 2D plotting function
    
    Parameters
    ----------
    x: array-like
        A 2-dimensional array containing the data to be plotted on the x axis
    y: array-like
        A 2-dimensional array containing the data to be plotted on the y-axis
    xyLabels: list
        The x and y axis labels in a list of length 2
    types: list or string
        The type of plot to correlating each set of data, can be "p" or "s" for plot or scatter
        If only one is entered all plots will follow the entered type
    labels: optional, list
         The labels for each dataset, the default is None
    xExtents: optional, array-like
        The upper and lower bound for the x-axis, the default is None
    yExtents: optional, array-like
        The upper and lower bound for the y-axis, the default is None
    grid: optional, boolean
        Whether to plot a grid, the default is True
    title: optional, string
        The title for the plot, the default is None
    sharex: optional, boolean
        Whether to use just one dataset for x, the default is False
    thickness: optional, float
        The linethickness or point size for the plot, the default is 1
    yScale: optional, string
        The type of scale to use on the y-axis, the default is "linear"
    xScale: optional, string
        The type of scale to use on the x-axis, the default is "linear"
    invertY: optional, boolean
        Whether to plot the y axis as descending, the default is False
    yTicks: optional, array-like
        The option to manually set the ticks on the y-axis
    
    Returns
    ----------
    Returns: The 2D plot of the data
    """
    
    # Creating the subplot
    fig, ax = plt.subplots()
    
    # Checking if the labels were entered as a list, creating a list if not and if no labels entered
    if type(labels) == str or labels == None:
        labelVal = labels
        labels = []
        for i in range(len(y)):
            labels.append(labelVal)
    
    # Checking if the types were entered as a list, creating a list if not
    if type(types) == str:
        typeVal = types
        types = []
        for i in range(len(y)):
            types.append(typeVal)
    
    # Allowing the input of one x array if desired
    if sharex == True:
        xs=[x]
        for i in range(len(y)):
            xs.append(x)
        x = xs
            
    # Looping through the data to be plot and checking the plottype
    for i in range(len(y)):
        
        # Plotting as a line plot
        if types[i] == 'p':
            ax.plot(x[i], y[i], label = labels[i], lw=thickness, ls=line[i], color="k")
        
        # Plotting as a scatter plot
        elif types[i] == 's':
            ax.scatter(x[i], y[i], s=thickness, label = labels[i])
        
    # Setting the axes below the data
    ax.set_axisbelow(True)
    
    # Labelling the axes
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    
    # Setting the title
    ax.set_title(title)
    
    # Setting the axis scale types
    ax.set_yscale(yScale)
    ax.set_xscale(xScale)
    
    ax.set_aspect(aspect)
    
    # 
    if yTicks != None:
        ax.set_yticks(yTicks[0], yTicks[1])
    
    # Checking whether to plot a grid
    if grid == True:
        ax.grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
        ax.grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
        ax.minorticks_on()
    
    # Setting the plotting extents if they are specified
    if xExtents != None:
        ax.set_xlim(xExtents[0], xExtents[1])
    if yExtents != None:
        ax.set_ylim(yExtents[0], yExtents[1])
        
    # Applying legend if true
    if labels[0] != None:
        ax.legend(markerscale = 10)
    
    # Inverting the direction of the y-axis if True
    if invertY == True:
        plt.gca().invert_yaxis() 
    
    plt.show()

def subplotting(x, y, xylabels, labels=None, xExtents=None, yExtents=None, grid = True, 
                figsize = (7,5), title=None):
    
    # Creating the figure and axes objects
    fig, axs = plt.subplots(len(x), figsize = figsize, sharex=True)
    
    # Looping through each x value to determine the number of subplots
    for i in range(len(x)):
        lines = []
        
        # Looping through the values within each subplot
        for n in range(len(x[i])):
            
            # Plotting the data and saving it as a line for the legend
            line, = axs[i].plot(x[i][n], y[i][n], color="k")
            lines.append(line)
        
        # Setting the y label of the subplot
        axs[i].set_ylabel(xylabels[1][i])
        
        # applying the grid
        if grid == True:
            axs[i].grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
            axs[i].grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
            axs[i].minorticks_on()
        
        # Setting the plotting extents if they are specified
        if xExtents != None:
            axs[i].set_xlim(xExtents[i][0], xExtents[i][1])
        if yExtents != None:
            axs[i].set_ylim(yExtents[i][0], yExtents[i][1])
        
        # Applying legend if true
        if labels != None:
            axs[i].legend(lines, labels)
    
    # applying a title and the x axis label
    axs[0].set_title(title)
    axs[len(axs)-1].set_xlabel(xylabels[0])
    
    fig.tight_layout()    

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
    
# =============================================================================
# Integration and Rotation
# =============================================================================
    
def rotation(time, aX, aY, aZ, gX, gY, gZ):
    
    #flipping signs
    # gX, gY, gZ = -gX, -gY, -gZ
    
    # finding the respective rotational angles in radians
    # t, rX, rY, rZ = np.deg2rad(integrationTrapezoid(time, gX, gY, gZ))
    t, rX, rY, rZ = np.deg2rad(euler(time, gX, gY, gZ, 0.01))
    
    # setting up acceleration vectors
    rAX = []
    rAY = []
    rAZ = []
    
    # Looping through all of the acceleration values and rotating based on their respective angular velocities
    for i in range(len(aX)):
        accVec = []
        accVec = np.array([aX[i], aY[i], aZ[i]])
        
        # Defining the rotation matrix
        rotMatX = np.array([[1, 0, 0], [0, np.cos(rX[i]), -np.sin(rX[i])], [0, np.sin(rX[i]), np.cos(rX[i])]])
        rotMatY = np.array([[np.cos(rY[i]), 0, np.sin(rY[i])], [0, 1, 0], [-np.sin(rY[i]), 0, np.cos(rY[i])]])
        rotMatZ = np.array([[np.cos(rZ[i]), -np.sin(rZ[i]) ,0], [np.sin(rZ[i]), np.cos(rZ[i]), 0], [0, 0, 1]])
        
        # Rotating the x, y, and z accelerations
        rotatedX = np.matmul(rotMatX, accVec)
        rotatedY = np.matmul(rotMatY, rotatedX)
        rotatedZ = np.matmul(rotMatZ, rotatedY)
        
        # appending to the new array
        rAX.append(rotatedZ[0])
        rAY.append(rotatedZ[1])
        rAZ.append(rotatedZ[2])
        
    return(time,rAX,rAY,rAZ)

def integrationTrapezoid(time,X,Y,Z):
    
    # Integrating the input x, y and z values
    XI = scI.cumulative_trapezoid(X,time, initial=0)
    YI = scI.cumulative_trapezoid(Y,time, initial=0)
    ZI = scI.cumulative_trapezoid(Z,time, initial=0)
    
    return(time, XI, YI, ZI)

def euler(time, X, Y, Z, dt):
    
    # Initialising the arrays for the integrated data with a zero to keep the same length
    XI, YI, ZI= [0], [0], [0]
    
    # Looping through the input data and integrating using euler's method
    for i in range(len(X)):
        XI.append(XI[i]+(X[i] * dt))
        YI.append(YI[i]+(Y[i] * dt))
        ZI.append(ZI[i]+(Z[i] * dt))
        
    return(time, np.array(vx), np.array(vy), np.array(vz))

# =============================================================================
# Calibration and Filtering
# =============================================================================

def savgolFilter(x, y, z, window=80, Order=9):

    # Applying the filter to each axis
    fX = savgol_filter(ax, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fY = savgol_filter(ay, window_length = int(window), polyorder = int(Order), mode = 'interp')
    fZ = savgol_filter(az, window_length = int(window), polyorder = int(Order), mode = 'interp')

    return(fX,fY,fZ)

def butterFilter(time, x, y, z, freq, fType):
    
    # Defining the samplerate of the data and the type of filter to use
    rate = len(time)/time.max()
    Pass = signal.butter(5, freq, fType, fs = rate, output = "sos")
    
    # applying the filter to each axis
    filteredSigx = signal.sosfilt(Pass, x)
    filteredSigy = signal.sosfilt(Pass, y)
    filteredSigz = signal.sosfilt(Pass, z)
    
    return filteredSigx, filteredSigy, filteredSigz

def dataCalibration(dataframe, inTime):
    
    # taking just the values below the inputted time
    dataframeInTime = dataframe[dataframe.Time <= inTime].copy()
    lowerUpperMeans = []
    
    # Looping through each variable in the dataframe
    for column in dataframe.drop("Time", axis=1).columns:
        
        # Taking the mean and standard deviation of the column
        mean = dataframeInTime[column].mean()
        std = dataframeInTime[column].std()
        
        # Translating the column by the mean and taking lower and upper means
        dataframe[column] = dataframe[column] - mean
        lowerMeans = dataframe[column] - (mean - std)
        upperMeans = dataframe[column] - (mean + std)
        
        lowerUpperMeans.append((column, lowerMeans, upperMeans))
        
    return dataframe, lowerUpperMeans
    
filePaths = ["DataDay1/rotZ1_25_02_2025_Data.txt"]
# filePaths = ["DataDay2SRFix/LshapeXY1_04_03_2025_Data.txt"]

for file in filePaths:
    
    # Reading the datafile and taking data from it
    dataF = pd.read_csv(file, sep='\\s+')
    t, ax, ay, az, gx, gy, gz = np.array(dataF["Time"]), dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    
    print(f"sample rate = {len(ax)/t.max()}")
    
    dataF, LUM = dataCalibration(dataF, 2)
    t, cAX, cAY, cAZ, cGX, cGY, cGZ = np.array(dataF["Time"]), dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    
# =============================================================================
#     Filtering the data with Butterworth
# =============================================================================

    # Butterworth
    freq = [10]
    ty = "lowpass"
    #ty = "highpass"
    #ty = "bandpass"
    fAX, fAY, fAZ = butterFilter(t, cAX, cAY, cAZ, freq, ty)
    fGX, fGY, fGZ = butterFilter(t, cGX, cGY, cGZ, freq, ty)
# =============================================================================
#     # Savitzky-Golay
#     fAX, fAY, fAZ = savgolFilter(cAX, cAY, cAZ)
#     fGX, fGY, fGZ = savgolFilter(cGX, cGY, cGZ)
# =============================================================================
     
    # Putting the Filtered Data Into a Dataframe
    filteredDataF = pd.DataFrame(np.array((t, fAX, fAY, fAZ, fGX, fGY, fGZ)).T, columns = dataF.columns)
    
    # Calibrating the Filtered Data
    dataF, LUM = dataCalibration(filteredDataF, 2)
    t, fAX, fAY, fAZ, fGX, fGY, fGZ = np.array(dataF["Time"]), dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]

# =============================================================================
#     Integrating the Data to Achieve Position and Velocity
# =============================================================================

    # Integrating the filtered acceleration data for exemplar data (raw velocity and position)
    t, vx, vy, vz = integrationTrapezoid(t, fAX, fAY, fAZ)
    t, x, y, z = integrationTrapezoid(t, vx, vy, vz)
    
    # Rotating the data to the lab reference frame
    t, rAX, rAY, rAZ = rotation(t, fAX, fAY, fAZ, fGZ, fGY, fGZ)
    
    # Integrating using cumulative trapezoid
    t, rVX, rVY, rVZ = integrationTrapezoid(t, rAX, rAY, rAZ)
    t, rX,rY,rZ = integrationTrapezoid(t, rVX, rVY, rVZ)
    
    # t, rVX, rVY, rVZ = euler(t, rAX, rAY, rAZ, 0.01)
    # t, rX, rY, rZ = euler(t, rVX, rVY, rVZ, 0.01)
     
# =============================================================================
#     Plotting
# =============================================================================
    
    # Index      = [0     , 1  , 2  , 3  , 4  , 5  , 6  , 7 , 8 , 9 ]
    
    rawData      = [t     , ax , ay , az , gx , gy , gz             ]
    calibData    = [t     , cAX, cAY, cAZ, cGX, cGY, cGZ            ]
    filteredData = [t     , fAX, fAY, fAZ, vx , vy , vz , x , y , z ]
    rotatedData  = [t     , rAX, rAY, rAZ, rVX, rVY, rVZ, rX, rY, rZ]
    axes         = ["Time (s)", "Acceleration (m/s$^2$)", "Velocity (m/s)", "Displacement (m)"]
    
    # Plotting the Path in 2D
    plotting([x, rX], [y, rY], ["x (m)", "y (m)"], "p", ["Non-Rotated", "Rotated"], aspect = "equal", line=["-","--"])
    
    # Plotting the Acceleration as a Function of Time to Demonstrate Filtering
    # plotting(t, [cAX, fAX], axes[0:2], "p", sharex=True, labels = ["Raw Data", "Filtered"], title = "x Filtered")
    # plotting(t, [cAY, fAY], axes[0:2], "p", sharex=True, labels = ["Raw Data", "Filtered"], title = "y Filtered")
    # plotting(t, [cAZ, fAZ], axes[0:2], "p", sharex=True, labels = ["Raw Data", "Filtered"], title = "z Filtered")
    
    # Plotting the Integration Steps as Subplots
    # subplotting([[t],[t],[t]], [[rAX], [rVX], [rX]], [axes[0], axes[1:4]])
    # subplotting([[t],[t],[t]], [[rAY], [rVY], [rY]], [axes[0], axes[1:4]], title="y Integration Steps")
    # subplotting([[t],[t],[t]], [[rAZ], [rVZ], [rZ]], [axes[0], axes[1:4]], title="z Integration Steps")

    # plotting(t, rotatedData[7:10], axes[0:2], "p", sharex=True, labels=["$x_R$", "$y_R$", "$z_R$"])
    