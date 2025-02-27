# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:25 2025

@author: Sam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scI

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

# =============================================================================
# def rotation(time, aX, aY, aZ, gX, gY, gZ):
#     #flipping signs
#     gX, gY, gZ = -gX, -gY, -gZ
#     
#     #finding the respective rotational angles in radians
#     radT = np.pi/180
#     #rX = scI.cumulative_trapezoid(gX, None, 0.01)*radT
#     #rY = scI.cumulative_trapezoid(gY, None, 0.01)*radT
#     #rZ = scI.cumulative_trapezoid(gZ, None, 0.01)*radT
#     rX = gX*time*radT
#     rY = gY*time*radT
#     rZ = gZ*time*radT    
# 
#     #setting up acceleration vectors
#     accVec = []
#     for i in range(len(aX)):
#         
#         accVec.append([aX[i], aY[i], aZ[i]])
#     
#     accVec = np.array(accVec)
#     
#     #setting up rotational matrices
#     rotMatX = np.array([[1, 0, 0], [0, np.cos(rX), -np.sin(rX)], [0, np.sin(rX), np.cos(rX)]])
#     rotMatY = np.array([[np.cos(rY), 0, np.sin(rY)], [0, 1, 0], [-np.sin(rY), 0, np.cos(rY)]])
#     rotMatZ = np.array([[np.cos(rZ), -np.sin(rZ) ,0], [np.sin(rZ), np.cos(rZ), 0], [0, 0, 1]])
#     
#     rotateByX = np.matvec(rotMatX, accVec)
#     rotateByY = np.matvec(rotMatY, rotateByX)
#     rotateByZ = np.matvec(rotMatZ, rotateByY)
#     
#     print(rotateByZ)
# =============================================================================
    
def rotation(time, aX, aY, aZ, gX, gY, gZ):
    #flipping signs
    gX, gY, gZ = -gX, -gY, -gZ
    
    #finding the respective rotational angles in radians
    # radT = np.pi/180
    # rX = scI.cumulative_trapezoid(gX, None, 0.01)*radT
    # rY = scI.cumulative_trapezoid(gY, None, 0.01)*radT
    # rZ = scI.cumulative_trapezoid(gZ, None, 0.01)*radT
    rX = np.deg2rad(gX)
    rY = np.deg2rad(gY)
    rZ = np.deg2rad(gZ)

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
        
        rotatedZ = np.matmul(rotMatZ, accVec)
        rotatedY = np.matmul(rotMatY, rotatedZ)
        rotatedX = np.matmul(rotMatX, rotatedY)
        
        rAX.append(rotatedX[0])
        rAY.append(rotatedX[1])
        rAZ.append(rotatedX[2])
    
    # rAZ = np.array(rAZ)-gravity
    
    plotting([time.iloc[1:],time.iloc[1:],time.iloc[1:]], [rAX,rAY,rAZ], ["Time (s)", "Acceleration (m/s$^2$)"], ['$a_y$','$a_y$','$a_z$'], ["pl", "pl", "pl"])
    
    return(time.iloc[1:],rAX,rAY,rAZ)

def integrationTrapezoid(time,X,Y,Z,dt):
    
    # Integrating the acceleration arrays to get velocity
    XI = scI.cumulative_trapezoid(X,time)
    YI = scI.cumulative_trapezoid(Y,time)
    ZI = scI.cumulative_trapezoid(Z,time)
    timeI = time[1:]
    
    return(XI, YI, ZI, timeI)

def plotting3D(x, y, z, axisLabels):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection="3d")
    
    ax.plot(x,y,z)
    ax.set_xlabel(axisLabels[0])
    ax.set_ylabel(axisLabels[1])
    ax.set_zlabel(axisLabels[2])

filePaths = ["DataDay1/xMoveTest1_25_02_2025_Data.txt"]
# filePaths = ["DataDay1/yMoveTest1_25_02_2025_Data.txt"]

for file in filePaths:
    
    dataF = pd.read_csv(file, sep='\\s+')
    #dataF = dataF.replace({"['inf']": np.nan})
    #dataF = dataF.dropna() 
    
    t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
    az=np.array(az)-9.81
    vx, vy, vz, vTime = integrationTrapezoid(t, ax, ay, az, 0.01)
    x,y,z, posTime = integrationTrapezoid(vTime, vx, vy, vz, 0.01)
    
    raTime, rAX, rAY, rAZ = rotation(t, ax, ay, az, gx, gy, gz)
    rVX, rVY, rVZ, rvTime = integrationTrapezoid(raTime, rAX, rAY, rAZ, 0.01)
    rX,rY,rZ, rposTime = integrationTrapezoid(rvTime, rVX, rVY, rVZ, 0.01)
    # change order idiot
    
    #plot results            
    plotting([t,t,t], [ax,ay,az], ["Time (s)", "Acceleration (m/s$^2$)"], ['$a_y$','$a_y$','$a_z$'], ["pl", "pl", "pl"])
    plotting([t,t,t], [gx,gy,gz], ["Time (s)", "Angular Velocity (deg/s)"], ['$\\omega_x$','$\\omega_y$','$\\omega_z$'], ["pl", "pl", "pl"])
    
    plotting3D(rX,rY,rZ, ["x","y","z"])
    
    # plotting([rposTime, posTime],[rX, x],["t","x"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([rposTime, posTime],[rY, y],["t","y"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([rposTime, posTime],[rZ, z],["t","z"],["Rotated", "Unrotated"],["pl", "pl"])
    
    # plotting([rvTime, vTime],[rVY, vy],["t","vy"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([rvTime, vTime],[rVX, vx],["t","vx"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([rvTime, vTime],[rVZ, vz],["t","vz"],["Rotated", "Unrotated"],["pl", "pl"])
    
    # plotting([raTime, t],[rAY, ay],["t","ay"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([raTime, t],[rAX, ax],["t","ax"],["Rotated", "Unrotated"],["pl", "pl"])
    # plotting([raTime, t],[rAZ, az],["t","az"],["Rotated", "Unrotated"],["pl", "pl"])
    
    print(f"The max acceleration is: {az.max()}\nThe max angular velocity is: {gz.max()}")
    
    func = 5*np.sin(t)
    intfunc = scI.cumulative_trapezoid(func,t)
    fig,ax=plt.subplots()
    ax.plot(t, func, label = "f")
    ax.plot(t[1:],intfunc-5, label= "fint")
    ax.legend()
    