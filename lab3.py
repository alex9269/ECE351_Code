# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:36:34 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#----------------------Part 1---------------------------------------

steps = 0.01
t = np.arange(0,20+steps,steps)

def step(t):
    y = np.zeros((len(t)))
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def ramp(t):
    y = np.zeros((len(t)))
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
             y[i] = t[i]
    return y


def func_1(t):
    y = step(t-2)-step(t-9)
    return y

def func_2(t):
    y = (np.exp(-t))*step(t)
    return y

def func_3(t):
    y = (ramp(t-2)*(step(t-2)-step(t-3)))+(ramp(4-t)*(step(t-3)-step(t-4)))
    return y

y = func_1(t)
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 1')
plt.title('Three Defined Functions')

y = func_2(t)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 2')

y = func_3(t)
plt.subplot(3,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 3')
plt.xlabel('t')
plt.show()

#------------------Part 2---------------------------------------------

def my_conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1,np.zeros((1,Nf2-1)))
    f2Extended = np.append(f2,np.zeros((1,Nf1-1)))
    result = np.zeros(f1Extended.shape)
    for i in range(Nf2+Nf1-2):
        result[i] = 0
        for j in range(Nf1):
            if ((i-j+1) > 0):
                try:
                    result[i]=result[i]+f1Extended[j]*f2Extended[i-j+1]
                except:
                    print(i,j)
    return result

t2 = np.arange(0,40+3*steps,steps)
y = my_conv(func_1(t),func_2(t))*steps
z = sig.convolve(func_1(t),func_2(t),'full')*steps
myFigSize = (8,4)
plt.figure(figsize=myFigSize)
plt.plot(t2,y,'r',label='my convolution')
plt.plot(t2,z,'--k',label='scipy convolution')
plt.legend(loc='upper right')
plt.grid(True)
plt.title('f1*f2')
plt.ylabel('y')
plt.xlabel('t')
plt.show()

y = my_conv(func_2(t),func_3(t))*steps
z = sig.convolve(func_2(t),func_3(t),'full')*steps
plt.figure(figsize=myFigSize)
plt.plot(t2,y,'r',label='my convolution')
plt.plot(t2,z,'--k',label='scipy convolution')
plt.legend(loc='upper right')
plt.grid(True)
plt.title('f2*f3')
plt.ylabel('y')
plt.xlabel('t')
plt.show()

y = my_conv(func_1(t),func_3(t))*steps
z = sig.convolve(func_1(t),func_3(t),'full')*steps
plt.figure(figsize=myFigSize)
plt.plot(t2,y,'r',label='my convolution')
plt.plot(t2,z,'--k',label='scipy convolution')
plt.legend(loc='upper right')
plt.grid(True)
plt.title('f1*f3')
plt.ylabel('y')
plt.xlabel('t')
plt.show()          

    
    