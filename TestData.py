# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:11:58 2025

@author: brada
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import signal
from scipy.signal import savgol_filter
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams["axes.formatter.limits"] = -2, 2

np.random.seed(0)

def plotting(x, y, xyLabels, labels, types, labelsOn = True, xExtents = None, yExtents = None, grid = True, title = None, sharex = False):
    
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
            ax.plot(x[i], y[i], label = labels[i], color="k")
            
        elif types[i] == 's':
            ax.scatter(x[i], y[i], s=5, label = labels[i])
            
        elif types[i] == 'l':
            ax.vlines(x[i], 0, 1, color='grey', zorder=0, linestyle = '-.', lw=0.5, label = labels[i])
    
    # Checking whether to plot a grid
    if grid == True:
        ax.grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
        ax.grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
        ax.minorticks_on()
    # ax.set_aspect(1)
        
    # Setting the axes below the data
    ax.set_axisbelow(True)
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    ax.set_title(title)
    
    # Setting the plotting extents if they are specified
    if xExtents != None:
        ax.set_xlim(xExtents[0], xExtents[1])
    if yExtents != None:
        ax.set_ylim(yExtents[0], yExtents[1])
    
    # Applying legend if true
    if labelsOn == True:
        ax.legend()
    
def fx(t):
    qt = len(t)//6
    
    x1 = np.zeros(500)
    x2 = np.sin(2*np.pi*t[2*qt:3*qt])
    x3 = np.zeros(qt)
    x4 = -np.sin(2*np.pi*t[4*qt:5*qt])
    x5 = np.zeros(qt)
    
    x = np.concatenate((x1,x2,x3,x4,x5))
    
    return x
    
def fy(t):
    qt = len(t)//6
    
    y1 = np.zeros(500)
    y2 = np.zeros(qt)
    y3 = np.sin(2*np.pi*t[4*qt:5*qt])
    y4 = np.zeros(qt)
    y5 = -np.sin(2*np.pi*t[2*qt:3*qt])
    
    y = np.concatenate((y1,y2,y3,y4,y5))
    
    return y

def integrationTrapezoid(time,X,Y,Z):
    
    XI = sc.integrate.cumulative_trapezoid(X,time, initial=0)
    YI = sc.integrate.cumulative_trapezoid(Y,time, initial=0)
    ZI = sc.integrate.cumulative_trapezoid(Z,time, initial=0)
    
    return(time, XI, YI, ZI)

def butterFilter(time, x, y, z, freq, fType):
    rate = len(time)/time.max()
    Pass = signal.butter(5, freq, fType, fs = rate, output = "sos")
    filteredSigx = signal.sosfilt(Pass, x)
    filteredSigy = signal.sosfilt(Pass, y)
    filteredSigz = signal.sosfilt(Pass, z)
    
    return filteredSigx, filteredSigy, filteredSigz

def subplotting(x, y, xylabels, labels=None, xExtents=None, yExtents=None, labelsOn=True, grid = True, figsize = (6,5), title=None):
    
    fig, axs = plt.subplots(len(x), figsize = figsize)
    for i in range(len(x)):
        lines = []
        for n in range(len(x[i])):
            
            line, = axs[i].plot(x[i][n], y[i][n], color="k")
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

def dataCalibration(time, ax, ay, az):
    
    calibIndices = (time<=2)
    
    variables = [ax[calibIndices], ay[calibIndices], az[calibIndices]]
    corrections = []
    
    for variable in variables:
        
        mean = variable.mean()
        std = variable.std()
        corrections.append([mean, std])
        
    return np.array(corrections)
axes         = ["Time (s)", "Acceleration (m/s$^2$)", "Velocity (m/s)", "Displacement (m)"]

time = np.linspace(0, 6, 1500)
ax = fx(time)
ay=fy(time)
az = np.zeros(len(time))
noise = np.random.normal(-0.1*ax.max(),0.1*ax.max(), 1500)
noise2 = np.random.normal(-0.3*ax.max(),0.3*ax.max(), 1500)

nax = ax + noise
nay = ay + noise
nax2 = nax + noise2
nay2 = nay + noise2
naz2 = az
t, nvx2, nvy2, nvz2 = integrationTrapezoid(time, nax2, nay2, naz2)
t, nx2, ny2, nz2 = integrationTrapezoid(time, nvx2, nvy2, nvz2)
plotting([nx2],[ny2], ["x Position (m)","y Position (m)"], [None], "p", labelsOn=False)
subplotting([[t], [t], [t]], [[nax2], [nvx2], [nx2]], ["Time (s)", axes[1:4]], labelsOn = False)

nax, nay, naz = butterFilter(time, nax, nay, az, [15], "lowpass")
nax2, nay2, naz2 = butterFilter(time, nax2, nay2, az, [15], "lowpass")
print(np.mean(nax), np.mean(ax))
# nax, nay, naz = butterFilter(time, nax, nay, az, [5], "lowpass")
# nax, nay, naz = savgolFiltering(nax, nay, az)
print("1")

corrections = dataCalibration (time, nax, nay, naz)
dataArrays = [nax, nay, naz]
upperMeans = [[], [], [], [], [], []]
lowerMeans = [[], [], [], [], [], []]

for i in range(len(dataArrays)):
    dataArrays[i], lowerMeans[i], upperMeans[i] = correcting(dataArrays[i], corrections[i][0], corrections[i][1])

nax, nay, naz = dataArrays

t, nvx, nvy, nvz = integrationTrapezoid(time, nax, nay, naz)
t, nx, ny, nz = integrationTrapezoid(time, nvx, nvy, nvz)

corrections = dataCalibration (time, nax2, nay2, naz2)
dataArrays = [nax2, nay2, naz2]
upperMeans = [[], [], [], [], [], []]
lowerMeans = [[], [], [], [], [], []]

for i in range(len(dataArrays)):
    dataArrays[i], lowerMeans[i], upperMeans[i] = correcting(dataArrays[i], corrections[i][0], corrections[i][1])
    
t, vx, vy, vz = integrationTrapezoid(time, ax, ay, az)
t, x, y, z = integrationTrapezoid(time, vx, vy, vz)

nax2, nay2, naz2 = dataArrays
t, nvx2, nvy2, nvz2 = integrationTrapezoid(time, nax2, nay2, naz2)
t, nx2, ny2, nz2 = integrationTrapezoid(time, nvx2, nvy2, nvz2)


# plotting(time, [ax,ay], ["t", "d"], [None, None], "p", labelsOn = False, sharex = True)
plotting([x],[y], ["x Position (m)","y Position (m)"], [None], "p", labelsOn=False)
plotting([nx],[ny], ["x Position (m)","y Position (m)"], [None], "p", labelsOn=False)
plotting([nx2],[ny2], ["x Position (m)","y Position (m)"], [None], "p", labelsOn=False)
subplotting([[t], [t], [t]], [[ax], [vx], [x]], ["Time (s)", axes[1:4]], labelsOn = False)
subplotting([[t], [t], [t]], [[nax], [nvx], [nx]], ["Time (s)", axes[1:4]], labelsOn = False)
subplotting([[t], [t], [t]], [[nax2], [nvx2], [nx2]], ["Time (s)", axes[1:4]], labelsOn = False)



