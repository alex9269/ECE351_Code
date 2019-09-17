# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:59:58 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(0,10+steps,steps)
  
def func1(t):
    y = np.zeros((len(t),1))
      
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y
  
y = func1(t)
  
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y) 
plt.grid(True)
plt.ylabel('y')
plt.xlabel('x')
plt.title('y(t)=cos(t)')

steps = 0.1
t = np.arange(-5,10+steps,steps)

def step(t):
    y = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y



def ramp(t):
    y = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
             y[i] = t[i]
    return y

def my_func(t):
    y = np.zeros((len(t),1))
    
    y = ramp(t-0) + 5*step(t-3) - ramp(t-3) - 2*step(t-6) - 2*ramp(t-6)
    
    return y


y = step(t)
myFigSize = (10,20)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Step Function')
plt.title('Step, Ramp, and User-Defined Functions')

y = ramp(t)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Ramp Function')

y = my_func(t)
plt.subplot(3,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('User-Defined Function')
plt.xlabel('t')
plt.show()

t = np.arange(-10,5+steps,steps)
y = my_func(-t)
myFigSize = (10,40)
plt.figure(figsize=myFigSize)
plt.subplot(6,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Time Reversal')
plt.title('Shifted User-Defined Function Graphs')

t = np.arange(-1,15+steps,steps)
y = my_func(t-4)
plt.subplot(6,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('t-4')

t = np.arange(-14,1+steps,steps)
y = my_func(-t-4)
plt.subplot(6,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('-t-4')

t = np.arange(-5,20+steps,steps)
y = my_func(t/2)
plt.subplot(6,1,4)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('t/2')

t = np.arange(-2,5+steps,steps)
y = my_func(2*t)
plt.subplot(6,1,5)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('2t')

t = np.arange(-5,10+steps,steps)
dt = np.diff(t)
y = np.diff(my_func(t),axis=0)/dt
plt.subplot(6,1,6)
plt.plot(t[0:len(t)-1],y)
plt.grid(True)
plt.axis([-5,10,-10,20])
plt.ylabel('First Derivative')
plt.xlabel('t')
plt.show()