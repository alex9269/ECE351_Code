# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:55:43 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#----------------------Part 1---------------------------------------

steps = 0.001
t = np.arange(0,2+steps,steps)

def step(t):
    y = np.zeros((len(t)))
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def trans(t):
    y = (0.5-0.5*np.exp(-4*t)+np.exp(-6*t))*step(t)
    return y

num = [1,6,12]
den = [1,10,24]

tout, yout = sig.step((num,den),T=t)


y = trans(t)
myFigSize = (8,10)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Hand-Defined')
plt.title('Step Responses')

plt.subplot(2,1,2)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('Scipy')
plt.xlabel('t')
plt.show()

num2 = [1,6,12]
den2 = [1,10,24,0]

[R,P,_] = sig.residue(num2,den2)

print('R='+str(R))
print('P='+str(P))

#----------------------Part 2---------------------------------------
print('-----------------Part 2--------------------')

num3 = [25250]
den3 = [1,18,218,2036,9085,25250,0]

[R1,P1,_] = sig.residue(num3,den3)

print('R='+str(R1))
print('P='+str(P1))

def cos(r,p,t):
    y1 = np.zeros((len(t)))
    
    for i in range(len(r)):
        a = p[i].real
        w = p[i].imag
        k = abs(r[i])
        b = np.angle(r[i])
        y1 = (k*np.exp(a*t)*np.cos(w*t+b))*step(t) + y1
        
    return y1


num4 = [25250]
den4 = [1,18,218,2036,9085,25250]

tout2,yout2 = sig.step((num4,den4),T=t)


y1 = cos(R1,P1,t)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y1)
plt.grid(True)
plt.ylabel('sig.residue')
plt.title('Step Responses part 2')

plt.subplot(2,1,2)
plt.plot(tout2,yout2)
plt.grid(True)
plt.ylabel('sig.step')
plt.xlabel('t')
plt.show()

