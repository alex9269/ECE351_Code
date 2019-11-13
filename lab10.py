# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:21:25 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

#---------------------Part 1----------------------------------------------------

steps = 100
w = np.arange(1e3,1e6+steps,steps)

R = 1000
L = 0.027
C = 100e-09
B = 1/(R*C)
A = 1/(L*C)

mag_H = (1/(R*C)*w)/np.sqrt((1/(L*C)-w**2)**2+(1/(R*C)*w)**2)
phi_H = np.pi/2-np.arctan((1/(R*C)*w)/(1/(L*C)-w**2))
for i in range(len(w)):
    if (1/(L*C)-w[i]**2)<0:
        phi_H[i] -= np.pi

numH = [1/(R*C),0]
denH = [1,1/(R*C),1/(L*C)]

[w1,mag,phi] = sig.bode((numH,denH))


myFigSize=[7,8]
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.semilogx(w,20*np.log10(mag_H))
plt.ylabel('magnitude')
plt.title('Hand-Derived Bode Plots')
plt.grid(True)
plt.subplot(2,1,2)
plt.semilogx(w,phi_H*180/np.pi)
plt.ylabel('phase angle')
plt.xlabel('frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.semilogx(w1,mag)
plt.ylabel('magnitude')
plt.title('Scipy Bode Plots')
plt.grid(True)
plt.subplot(2,1,2)
plt.semilogx(w1,phi)
plt.ylabel('phase angle')
plt.xlabel('frequency')
plt.grid(True)
plt.show()

sys = con.TransferFunction(numH,denH)
_ = con.bode(sys,w,dB=True,Hz=True,deg=True,Plot=True)

#---------------------------Part 2--------------------------

fs = 10000000
steps1 = 1/fs
t = np.arange(0,0.01+steps1,steps1)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

[Z,P] = sig.bilinear(numH,denH,fs)

y = sig.lfilter(Z,P,x)

plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,x)
plt.ylabel('Signal')
plt.title('Signal Before and After Filtering')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(t,y)
plt.ylabel('Filtered Signal')
plt.xlabel('t')
plt.grid(True)
plt.show()




