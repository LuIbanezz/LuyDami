from cmath import pi
from math import sqrt
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def CalculateSens(H, R): #Calculates sensitivity of H respect of R
    
    sens = sp.diff(H, R) * R / H
    return sens

n2,n0,w02,Q,H,w0Q = sp.symbols('n2,n0,w02,Q,H,w0Q')
w = sp.symbols('w', real=True)
#w = 2*pi*14000 #Evalúo w en la frecuencia de la banda pasante
C3,C22,C21,R1,R42,R41,RA2,RA1,RB = sp.symbols('C3,C22,C21,R1,R42,R41,RA2,RA1,RB', real=True)
#Ga1, Ga2, Gb, G41, G42, G1  = sp.symbols('Ga1, Ga2, Gb, G41, G42, G1', real=True)

""" 
n2 = (Ga1 + Ga2+Gb)/Gb * (C22/(C21+C22)) - Ga2/Gb
n0 = G1/C3 * (G41 + G42)/(C21 + C22) * ((G42/(G41 + G42)) * ((Ga1 + Ga2)+Gb)/Gb - Ga2/Gb)
w02 = G1/C3 * (G41 + G42) / (C21 + C22)
w0Q = (G41 + G42)*(1/(C21+C22) + 1/C3) - (G1/(C21 + C22))*((Ga1 + Ga2)/Gb)
"""
 
n2 = ((1/RA1) + (1/RA2)+(1/RB))/(1/RB) * (C22/(C21+C22)) - (1/RA2)/(1/RB)
n0 = (1/R1)/C3 * ((1/R41) + (1/R42))/(C21 + C22) * (((1/R42)/((1/R41) + (1/R42))) * (((1/RA1) + (1/RA2))+(1/RB))/(1/RB) - (1/RA2)/(1/RB))
w02 = (1/R1)/C3 * ((1/R41) + (1/R42)) / (C21 + C22)
w0Q = ((1/R41) + (1/R42))*(1/(C21+C22) + 1/C3) - ((1/R1)/(C21 + C22))*(((1/RA1) + (1/RA2))/(1/RB))
 
H = (-n2*w**2 + n0)/(-w**2 + sp.I*w*w0Q + w02)

#Ga1, Ga2, Gb, G41, G42, G1 = 1/RA1, 1/RA2, 1/RB, 1/R41, 1/R42, 1/R1 

Gain = ( (sp.re(H))**2 + (sp.im(H))**2 )**(1/2)

print('La ganancia es:', Gain,'\n\n\n')

print('S^G_Ra1 = ', CalculateSens(Gain, RA1))
print('S^G_Ra2 = ', CalculateSens(Gain, RA2).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_R41 = ', CalculateSens(Gain, R41).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_R42 = ', CalculateSens(Gain, R42).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_R1 = ', CalculateSens(Gain, R1).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_Rb = ', CalculateSens(Gain, RB).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_C21 = ', CalculateSens(Gain, C21).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_C22 = ', CalculateSens(Gain, C22).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))
print('S^G_C3 = ', CalculateSens(Gain, C3).subs( {C3: 13E-09,C22: 12E-09, \
    C21: 1e-09,R1: 371.73, R42: 34500 ,R41: 2495.8 ,RA2: 56150 , RA1: 13300,RB: 1000.0}))