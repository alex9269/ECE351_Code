# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:39:07 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig

fs = 100
steps = 1/fs
t = np.arange(0,2,steps)

def fft(fs,x):

    N = len(x) 
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2,N/2)*fs/N 
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted)
    
    return X_mag,X_phi,freq

def fft_clean(fs,x):

    N = len(x) 
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2,N/2)*fs/N 
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if X_mag[i] <= 1e-10:
            X_phi[i] = 0
    
    return X_mag,X_phi,freq

def bk(k):
    y = (2/(k*np.pi))*(1-np.cos(k*np.pi))
    return y

t1 = np.arange(0,16,steps)
T = 8
w = (2*np.pi)/T
N = 15
y = 0

for i in range(1,N+1):
    y = bk(i)*np.sin(i*w*t1) + y

x = [np.cos(2*np.pi*t),
     5*np.sin(2*np.pi*t),
     2*np.cos(2*np.pi*2*t-2)+(np.sin(2*np.pi*6*t+3))**2,
     np.cos(2*np.pi*t),
     5*np.sin(2*np.pi*t),
     2*np.cos(2*np.pi*2*t-2)+(np.sin(2*np.pi*6*t+3))**2,
     y]


for i in range(len(x)):
    if i < 3:
        [X_mag,X_phi,freq] = fft(fs,x[i])
    else:
        [X_mag,X_phi,freq] = fft_clean(fs,x[i])
    
    myFigSize = (9,10)
    plt.figure(figsize=myFigSize)
    plt.subplot(3,1,1)
    if i==6:
        plt.plot(t1,x[i])
    else:
        plt.plot(t,x[i])
    plt.grid(True)
    plt.ylabel('y')
    plt.xlabel('t')
        
    plt.subplot(3,2,3)
    plt.stem(freq,X_mag,use_line_collection=True)
    plt.grid(True)
            
    plt.subplot(3,2,4)
    plt.stem(freq,X_mag,use_line_collection=True)
    plt.grid(True)
    if i==2 or i==5:
        plt.xlim(-15,15)
    else:
        plt.xlim(-2,2)
        
    plt.subplot(3,2,5)
    plt.stem(freq,X_phi,use_line_collection=True)
    plt.grid(True)
        
    plt.subplot(3,2,6)
    plt.stem(freq,X_phi,use_line_collection=True)
    plt.grid(True)
    if i==2 or i==5:
        plt.xlim(-15,15)
    else:
        plt.xlim(-2,2)
    plt.show()

