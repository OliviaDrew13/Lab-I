# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:25 2025

@author: Sam
"""

import pandas as pd
import matplotlib.pyplot as plt

filePath = "DataDay1/trialFile2_25_02_2025_Data.txt"
dataF = pd.read_csv(filePath, sep='\\s+')
#dataF = dataF.replace({"['inf']": np.nan})
#dataF = dataF.dropna()

print(dataF)
t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
#plot results            
plt.figure(1)
plt.subplot(211)
plt.plot(t,ax,'r',t,ay,'g',t,az,'b')
plt.ylabel('Acceleration (m/s$^2$)')
plt.legend(['$a_x$','$a_y$','$a_z$'],bbox_to_anchor=(1.0,1.0))
    
plt.subplot(212)
plt.plot(t,gx,'pink',t,gy,'yellow',t,gz,'cyan')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend(['$\\omega_x$','$\\omega_y$','$\\omega_z$'],bbox_to_anchor=(1.0,1.0))

plt.tight_layout()    

plt.show()
