from cmath import pi
from math import sqrt
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def CalculateSens(H, R): #Calculates sensitivity of H respect of R
    
    sens = sp.diff(H, R) * R / H
    return sens

H = sp.symbols('H')
#w = 2*pi*14000 #Evalúo w en la frecuencia de la banda pasante
w = sp.symbols('w', real=True)
R, C = sp.symbols('R, C', real= True)

H = (sp.I * w) / ( sp.I * w + 1 / (R * C) )

fp = (1) / (R*C)

Gain = ( (sp.re(H))**2 + (sp.im(H))**2 )**(1/2)

print('La ganancia es:', Gain,'\n\n\n')

print('S^G_R = ', CalculateSens(Gain, R))

print('S^G_C = ', CalculateSens(Gain, C))

print('S^fp_R = ', CalculateSens(fp, R))

print('S^fp_C = ', CalculateSens(fp, C))