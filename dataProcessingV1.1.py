# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:25 2025

@author: Sam
"""

import pandas as pd
import matplotlib.pyplot as plt

filePath = "DataDay1/gXTest2_25_02_2025_Data.txt"
dataF = pd.read_csv(filePath, sep='\\s+')
#dataF = dataF.replace({"['inf']": np.nan})
#dataF = dataF.dropna() 

print(dataF)
t, ax, ay, az, gx, gy, gz = dataF["Time"], dataF["ax"], dataF["ay"], dataF["az"], dataF["gx"], dataF["gy"], dataF["gz"]
#plot results            
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(t,ax,'r',t,ay,'g',t,az,'b')
axs[0].set_ylabel('Acceleration (m/s$^2$)')
axs[0].legend(['$a_x$','$a_y$','$a_z$'],bbox_to_anchor=(1.0,1.0))
    
axs[1].plot(t,gx,'pink',t,gy,'yellow',t,gz,'cyan')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Angular Velocity (deg/s)')
axs[1].legend(['$\\omega_x$','$\\omega_y$','$\\omega_z$'],bbox_to_anchor=(1.2,1.0))
# axs[0].set_xlim(4,6)


plt.tight_layout()    

plt.show()
