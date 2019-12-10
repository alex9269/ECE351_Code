# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:34:34 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import pandas as pd
import control as con

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

#-------------original figure
plt.figure(figsize=(10,7))
plt.plot(t,sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.show()

def make_stem (ax,x,y,color='k',style='solid',label='',linewidths=2.5,**kwargs):
    ax.axhline(x[0],x[-1],0,color='r')
    ax.vlines(x,0,y,color=color,linestyles=style,label=label,linewidths=linewidths)
    ax.set_ylim([1.05*y.min(),1.05*y.max()])

def fft_clean(fs,x):
    N = len(x) 
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2,N/2)*fs/N 
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if X_mag[i] <= 0.05:
            X_phi[i] = 0
    
    return X_mag,X_phi,freq

#--------------------------------first round of ffts
fs = 1e6
[X_mag,X_phi,freq] = fft_clean(fs,sensor_sig)

fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(8,15))
plt.subplot(ax1)
make_stem(ax1,freq,X_mag)
plt.grid(True)
plt.xscale('log')
plt.xlim(10**0,10**6)
plt.ylabel('Magnitude')
plt.title('Initial ffts')

plt.subplot(ax2)
make_stem(ax2,freq,X_mag)
plt.grid(True)
plt.xscale('log')
plt.xlim(5e1,7e1)
plt.ylabel('Lower Noise')

plt.subplot(ax3)
make_stem(ax3,freq,X_mag)
plt.grid(True)
plt.xscale('log')
plt.xlim(1e3,4e3)
plt.ylabel('Input Signal')

plt.subplot(ax4)
make_stem(ax4,freq,X_mag)
plt.grid(True)
plt.xscale('log')
plt.xlim(4e4,6e5)
plt.ylabel('Higher Noise')
plt.xlabel('Frequency (Hz)')
plt.show()

#-----------------------------Part 2-------------------------------------------

steps = 100
w = np.arange(1,1e6+steps,steps)

R = 200
L = 0.0125
C = 6e-07

mag_H = (1/(R*C)*w)/np.sqrt((1/(L*C)-w**2)**2+(1/(R*C)*w)**2)
phi_H = np.pi/2-np.arctan((1/(R*C)*w)/(1/(L*C)-w**2))
for i in range(len(w)):
    if (1/(L*C)-w[i]**2)<0:
        phi_H[i] -= np.pi
        
numH = [1/(R*C),0]
denH = [1,1/(R*C),1/(L*C)]


#-----------------------------------Bode plots of transfer function
plt.figure(figsize=(10,7))
sys = con.TransferFunction(numH,denH)
_ = con.bode(sys,w,dB=True,Hz=True,deg=True,Plot=True)

plt.figure(figsize=(10,7))
_=con.bode(sys,np.arange(5e1,7e1)*2*np.pi,dB=True,Hz=True,deg=True,Plot=True)

plt.figure(figsize=(10,7))
_=con.bode(sys,np.arange(1.78e3,2.2e3)*2*np.pi,dB=True,Hz=True,deg=True,Plot=True)


plt.figure(figsize=(10,7))
_=con.bode(sys,np.arange(4e4,6e5)*2*np.pi,dB=True,Hz=True,deg=True,Plot=True)

#----------------------------------------filtering noisy signal
[Z,P] = sig.bilinear(numH,denH,fs)

y = sig.lfilter(Z,P,sensor_sig)

plt.figure(figsize=(10,7))
plt.plot(t,y)
plt.title('Filtered Signal')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.show()

#--------------------------------------------filtered ffts
[X_mag2,X_phi2,freq2] = fft_clean(fs,y)

fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(8,15))
plt.subplot(ax1)
make_stem(ax1,freq2,X_mag2)
plt.grid(True)
plt.xscale('log')
plt.xlim(1e0,1e6)
plt.ylabel('Magnitude')
plt.title('Filtered ffts')

plt.subplot(ax2)
make_stem(ax2,freq2,X_mag2)
plt.grid(True)
plt.xscale('log')
plt.xlim(5e1,7e1)
plt.ylabel('Lower Noise')

plt.subplot(ax3)
make_stem(ax3,freq2,X_mag2)
plt.grid(True)
plt.xscale('log')
plt.xlim(1e3,4e3)
plt.ylabel('Input Signal')

plt.subplot(ax4)
make_stem(ax4,freq2,X_mag2)
plt.grid(True)
plt.xscale('log')
plt.xlim(4e4,6e5)
plt.ylabel('Higher Noise')
plt.xlabel('frequency (Hz)')
plt.show()









