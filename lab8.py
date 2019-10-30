# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:22:51 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def ak(k):
    if k==0:
        y=0
    else:
        y = (2/(k*np.pi))*np.sin(k*np.pi)
    return y

def bk(k):
    y = (2/(k*np.pi))*(1-np.cos(k*np.pi))
    return y

a = [ak(0),ak(1)]
print('a[0]=',a[0])
print('a[1]=',a[1])
b = [bk(1),bk(2),bk(3)]
print('b[1]=',b[0])
print('b[2]=',b[1])
print('b[3]=',b[2])

steps = 0.1
t = np.arange(0,20+steps,steps)
T = 8
w = (2*np.pi)/T
N = [1,3,15,50,150,1500]
myFigSize=[6,15]

for i in [1,2]:
    for j in ([1+(i-1)*3,2+(i-1)*3,3+(i-1)*3]):
        y=0
        for m in range(1,N[j-1]+1):
            y = bk(m)*np.sin(m*w*t) + y
        plt.figure(i,figsize=myFigSize)
        plt.subplot(3,1,(j-1)%3+1)
        plt.plot(t,y)
        plt.grid(True)
        plt.ylabel(f"N = {N[j-1]}")
        if j==1 or j==4:
            plt.title('Fourier Approximations')
        if j==3 or j==6:
            plt.xlabel('t')
            plt.show()



        
        
            
            