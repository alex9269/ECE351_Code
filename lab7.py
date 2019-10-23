# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:30:35 2019

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#----------------------Part 1---------------------------------------

numG = [1,9]
denG = [1,-2,-40,-64]

[ZG,PG,KG]= sig.tf2zpk(numG,denG)

print('ZG='+str(ZG))
print('PG='+str(PG))
print('KG='+str(KG))
print()

numA = [1,4]
denA = [1,4,3]

[ZA,PA,KA]= sig.tf2zpk(numA,denA)

print('ZA='+str(ZA))
print('PA='+str(PA))
print('KA='+str(KA))
print()

numB = [1,26,168]

ZB = np.roots(numB)

print('ZB='+str(ZB))
print()

numOL = [1,9]
denOL = sig.convolve(sig.convolve([1,3],[1,1]),sig.convolve([1,-8],[1,2]))
print('denOL='+str(denOL))
print()

tout, yout = sig.step((numOL,denOL))

myFigSize = (7,6)
plt.figure(figsize=myFigSize)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('y')
plt.title('Open Loop Step Response')
plt.xlabel('t')
plt.show()


#----------------------Part 2---------------------------------------


numCL = sig.convolve(numA,numG)
print('numCL='+str(numCL))
denCL = sig.convolve(sig.convolve(numB,numG)+denG,denA)
print('denCL='+str(denCL))
print()

[Zc,Pc,Kc] = sig.tf2zpk(numCL,denCL)

print('Zc='+str(Zc))
print('Pc='+str(Pc))
print('Kc='+str(Kc))
print()

tout1, yout1 = sig.step((numCL,denCL))

myFigSize = (7,6)
plt.figure(figsize=myFigSize)
plt.plot(tout1,yout1)
plt.grid(True)
plt.ylabel('y')
plt.title('Closed Loop Step Response')
plt.xlabel('t')
plt.show()

