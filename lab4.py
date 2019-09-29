# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:55:30 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#----------------------Part 1---------------------------------------

steps = 0.01
t = np.arange(-10,10+steps,steps)

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

def h1(t):
    y = (np.exp(2*t)*step(1-t))
    return y

def h2(t):
    y = step(t-2)-step(t-6)
    return y

f=0.25
w=2*np.pi*f
def h3(t):
    y = (np.cos(w*t)*step(t))
    return y

y = h1(t)
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 1')
plt.title('Three Defined Functions')

y = h2(t)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 2')

y = h3(t)
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

def f(t):
    y = step(t)
    return y

def y1(t):
    y = 0.5*((np.exp(2*t)*step(1-t))+(np.exp(2)*step(t-1)))
    return y

def y2(t):
    y = ramp(t-2)-ramp(t-6)
    return y

def y3(t):
    y = (1/w)*np.sin(w*t)*step(t)
    return y

t2 = np.arange(2*t[0],2*t[len(t)-1]+steps,steps)
y = my_conv(h1(t),f(t))*steps
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,y)
plt.grid(True)
plt.title('Convolutions')
plt.ylabel('h1*f')

y = my_conv(h2(t),f(t))*steps
plt.subplot(3,1,2)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('h2*f')

y = my_conv(h3(t),f(t))*steps
plt.subplot(3,1,3)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('h3*f')
plt.xlabel('t')
plt.show()


y = y1(t2)
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,y)
plt.grid(True)
plt.title('Hand-Derived Convolutions')
plt.ylabel('h1*f')

y = y2(t2)
plt.subplot(3,1,2)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('h2*f')

y = y3(t2)
plt.subplot(3,1,3)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('h3*f')
plt.xlabel('t')
plt.show()
 