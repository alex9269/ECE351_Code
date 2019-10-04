# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:51:41 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#----------------------Part 1---------------------------------------

steps = 0.00001
t = np.arange(0,0.0012+steps,steps)

def trans(t):
    p = (105*np.pi)/180
    y = 10356*np.exp(-5000*t)*np.sin(18584*t+p)
    return y

num = [0,1,0]
den = [0.0001,1,37037]

tout, yout = sig.impulse((num,den),T=t)

y = trans(t)
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Hand-Defined')
plt.title('Impulse Responses')

plt.subplot(2,1,2)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('Scipy')
plt.xlabel('t')
plt.show()

tout1, yout1 = sig.step((num,den),T=t)

myFigSize = (8,5)
plt.figure(figsize=myFigSize)
plt.plot(tout1,yout1)
plt.grid(True)
plt.ylabel('y')
plt.title('Step Response')
plt.xlabel('t')
plt.show()
