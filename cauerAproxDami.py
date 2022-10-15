from cmath import pi
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

N, Wn = signal.ellipord(14000*2*pi, 3500*2*pi, 1, 43, analog=True)
#print("El orden del filtro deberá ser ",N )

b, a = signal.ellip(N, 1, 43, Wn, 'high', analog=True, output='ba')

#print("La función transferencia es b/a donde\n b =", b, "y a = ",a )    

sos = signal.ellip(N, 1, 43, Wn, 'high', analog=True, output='sos')

#print("A continuacion expreso la funcion transferencia directamente como suma de funciones de segundo orden")
#print(sos)

#A partir de estos valores ya puede graficarse la respuesta en frecuencia con las siguiente linea

w, h = signal.freqs(b, a, np.logspace(3, 5, 500))
plt.semilogx(w/(2*pi), 20 * np.log10(abs(h)))
plt.title('Elliptical highpass filter fit to constraints')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.axis([1000, 20000, -180, 3])
#plt.show()

#
# Función transferencia 1 para realizar con Sedra
#

#((s**2 + 819315560)/(s**2 + 40559.8221 * s + 7713242650))

##Ecuaciones generales

w_0, Q, Q_0, G, C, G_1, G_b = sp.symbols('w_0,Q,Q_0,G,C,G_1,G_b')

w_0 = 7713242650**(1/2) # 87825,068
w_z = 819315560**(1/2)
Q = w_0/40559.8221
Q_0 = 1
C = 1E-9
eq_4 = sp.Eq( C, (G * w_0)/( 2 * Q_0))  

## G

eqs_1 = [eq_4]
sol_Q = (sp.solve(eqs_1, [G, Q]))[Q]
Q = sol_Q

print('Q = ', sol_Q)

## G_1. Tengo la opción de calcularlo más adelante
#eq_5 = sp.Eq(G_1, 4 * Q_0 ** 2 * G)
#solG_1 = (sp.solve(eq_5, [G_1]))[0]
#solG_1 = solG_1.subs( {Q_0:1, G: 1/(100E6)} )
#print( '1/G_1 = ', (1/solG_1))

## k, K, n, m, C22, C21

k,n,m,w_z,K = sp.symbols('k,n,m,w_z,K') 
eq_6 = sp.Eq(k, ((w_z/w_0)**2 / (1-Q_0/Q))) #Tomando n_2 = 1
eq_7 = sp.Eq(n, k * (1-(Q_0/(K*Q))))
eq_8 = sp.Eq(m,k*((K-1)/K)*(1+2*Q_0**2*(w_0/w_z)**2))
eq_9 = sp.Eq(K, 1+1/(2*Q_0**2)*(1-Q_0/Q))

C22, C21 = sp.symbols('C22, C21')
eq_10 = sp.Eq(C22/C21, m / (1 - m))
eq_11 = sp.Eq(C22 + C21, C)

eqs_2 = [eq_6,eq_7,eq_8,eq_9,eq_10,eq_11,]

sol_2 = sp.solve(eqs_2 , [k,m,n,K,C22,C21])

sol_k = sol_2[0][0].subs({w_z: 819315560**(1/2)})
sol_m = sol_2[0][1].subs({w_z: 819315560**(1/2)})
sol_n = sol_2[0][2].subs({w_z: 819315560**(1/2)})
sol_K = sol_2[0][3].subs({w_z: 819315560**(1/2)})
sol_C22 = sol_2[0][4].subs({w_z: 819315560**(1/2)})
sol_C21 = sol_2[0][5].subs({w_z: 819315560**(1/2)})

print('k = ', sol_k)
print('n = ', sol_n)
print('m = ', sol_m)
print('K = ', sol_K)
print('m/(1 - m)= ', sol_m / (1 - sol_m))
print('C22 = ', sol_C22)
print('C21 = ', sol_C21)

C22 = sol_C22
C21 = sol_C21

# W_p, Q_p, G_1

w_p, Q_p, w_t = sp.symbols('w_p, Q_p, w_t')

eq_12 = sp.Eq(w_p, w_0*(1+Q_0*(w_0/w_t)))
eq_13 = sp.Eq(G_1, 2*Q_0*w_p*(C*(C22+C21))**0.5)
eq_14 = sp.Eq(Q_p, Q*(1-2*Q_0*Q*(w_0/w_t)*(1/(2*Q)-(w_0/w_t))))

sol_3 = sp.solve([eq_12, eq_13, eq_14], [w_p, Q_p, G_1])

sol_w_p = sol_3[w_p].subs({w_t: 17.17571E6})
sol_Q_p = sol_3[Q_p].subs({w_t: 17.17571E6})
sol_G_1 = sol_3[G_1].subs({w_t: 17.17571E6})

print('w_p = ', sol_w_p)
print('Q_p = ', sol_Q_p)
print('G_1 = ', sol_G_1) 

Q_p = sol_Q_p
w_p = sol_w_p

G_b = 1E-3  #Número arbitrario a elección

## G_41, G_42, Ga_1, Ga_2

G_41, G_42, Ga_1, Ga_2 = sp.symbols('G_41, G_42, Ga_1, Ga_2')

eq_17 = sp.Eq((Ga_1+Ga_2)/G_b, ((G_41+G_42)/G_1)*((C21+C22+C)/C)-(w_p*(C21+C22)/(Q_p*G_1)))
eq_18 = sp.Eq( (((G_1*(G_41+G_42))/(C*(C21+C22)))*(((G_42/(G_41+G_42))*((Ga_1+Ga_2+G_b)/G_b))-(Ga_2/G_b)))/(((Ga_1+Ga_2+G_b)/(G_b))*((C22)/(C22+C21)) - (Ga_2/G_b)) , w_z**2)

sol_4 = sp.solve([eq_17,eq_18] , [G_42, Ga_2])
sol_G_42 = sol_4[0][0].subs({G_1 : sol_G_1, w_z: 819315560**(1/2)})
sol_Ga_2 = sol_4[0][1].subs({G_1 : sol_G_1, w_z: 819315560**(1/2)})

eq_15 = sp.Eq( G_41+G_42,G_1/(4*Q_0**2))
eq_16 = sp.Eq( ((Ga_1 + Ga_2 + G_b)/(G_b)) * G_42 * ((1/(C21+C22)) + 1/C) - (Ga_2/G_b)*((G_1 /(C21 + C22)) + (G_41 + G_42) *((1/(C21+C22)) + 1/C)), 0)
eq_19 = sp.Eq( G_42, sol_G_42)
eq_20 = sp.Eq( Ga_2, sol_Ga_2)

sol_5 = sp.solve( [eq_15,eq_16,eq_19,eq_20] , [G_41, G_42, Ga_1, Ga_2])

print(sol_5)

sol_G_41 = sol_5[0][0].subs({G_1 : sol_G_1})
sol_G_42 = sol_5[0][1].subs({G_1 : sol_G_1})
sol_Ga_1 = sol_5[0][2].subs({G_1 : sol_G_1})
sol_Ga_2 = sol_5[0][3].subs({G_1 : sol_G_1})

print('Ga_1 = ', sol_Ga_1)
print('G_41 = ', sol_G_41)
print('G_42 = ', sol_G_42)
print('Ga_2 = ', sol_Ga_2) 

#Valores finales de las resistencias

print('A continuación los valores finales que hay que poner en el circuito \n\n\n\n')

print('R1 = ', 1 / (sol_G_1))
print('Ra1 = ', 1 / (sol_Ga_1))
print('Ra2 = ', 1 / (sol_Ga_2))
print('Rb = ', 1 / (G_b))
print('G41 = ', (sol_G_41))
print('G42 = ', (sol_G_42))
print('R41 = ', 1 / (sol_G_41))
print('R42 = ', 1 / (sol_G_42))
print('C3 =' , C)
print('C22 =' , C22)
print('C21 =' , C21)