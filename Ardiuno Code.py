## GY521 Exp I Script ##
## (Partially) rewritten M. Bird 2023. Original Script B. Rackauskas ##
#edited by Olivia Drew and Bradley Day

import serial
import time

#establish serial connection to Arduino/GY521
ser = serial.Serial('COM5', 38400) #Baud rate 38400 Hz, COM port must match.

for i in range(0,3):
    #Inital lines from GY521, we can ignore them
    Junk = ser.readline(100)

time.sleep(1)
print('Connecting Device...')

res = 2**16; #16 bit resolution
a_sen = 2*9.81; #m/s^2 (intially 2g in Arduino Code)
g_sen = 250; #deg/s (intially 250 deg/s in Arduino Code)


Out = []
ax = []
ay = []
az = []
gx = []
gy = []
gz = []
t  = []

try:
    print("Capturing data, press ctrl+C to finish")    
    while True:
        Out.append(ser.readline()) #just read the data, we can decode it later
        

except KeyboardInterrupt:    
    ser.close()
    print('\nStopping... \nNow decoding output')
    
    
    for i in range(0,len(Out)):

        ss = Out[i].decode("utf-8","ignore").replace('\r\n','').split('\t')

        if ss[0] == 'a/g:': #now convert to "real" values e.g. +/- 2g divded by resolution
            ax.append(int(ss[1])*a_sen*2/res)
            ay.append(int(ss[2])*a_sen*2/res)
            az.append(int(ss[3])*a_sen*2/res)
            gx.append(int(ss[4])*g_sen*2/res)
            gy.append(int(ss[5])*g_sen*2/res)
            gz.append(int(ss[6])*g_sen*2/res)
            t.append(int(ss[7])/1000)
            
    print('Done')
    
    #writes to a text file
    timestr = time.strftime("%d_%m_%Y") 
    
    Filename = input("Enter Filename:")
    f = open(Filename + '_' + timestr + '_Data.txt','w')
    f.write('Time\tax\tay\taz\tgx\tgy\tgz\n')
    for i in range(len(ax)):
        f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (t[i],ax[i],ay[i],az[i],gx[i],gy[i],gz[i]))
    f.close()
    print('File written to:' + str(f))
    print('Done')
    