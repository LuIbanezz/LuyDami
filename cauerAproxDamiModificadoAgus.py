from cmath import pi
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Returns nominal value below to x
def inferior_nominal_value(x):
    n = 12*np.log10(x)
    nfloor = np.floor(n)
    res = 10 ** (nfloor/12)
    return res

def nearest_nominal_value(x):
    n = 12*np.log10(x)
    nround = np.round(n)
    res = 10 ** (nround/12)
    return res

N, Wn = signal.ellipord(14000*2*pi, 3500*2*pi, 0.5, 50, analog=True)
print("El orden del filtro deberá ser ",N )

b, a = signal.ellip(N, 0.5, 50, Wn, 'high', analog=True, output='ba')

print("La función transferencia es b/a donde\n b =", b, "y a = ",a )

sos = signal.ellip(N, 0.5, 50, Wn, 'high', analog=True, output='sos')

print("A continuacion expreso la funcion transferencia directamente como cascada de funciones de segundo orden")
print(sos)

# A partir de estos valores ya puede graficarse la respuesta en frecuencia con las siguiente linea

w, h = signal.freqs(b, a, np.logspace(3, 5, 500))
plt.semilogx(w/(2*pi), 20 * np.log10(abs(h)))
plt.title('Elliptical highpass filter fit to constraints')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.axis([1000, 20000, -180, 3])
#plt.show()

#Calculos para el RC
wrc = sos[0][5]
rc = 1/wrc
R = 220E3  
C = rc/R

#Calculos para el de ripple
hp = sos[1]
wt = 2 * np.pi * 3e6  # GBWP opamp
wz = np.sqrt(hp[2])     # w del cero
w0 = np.sqrt(hp[5])     # w del polo
Q = w0 / hp[4]

print('Q = ', Q)
print('w0 = ', w0)

# Paso 1
Q0 = 1.25
K = 1 + (1/(2*Q0**2))*(1-Q0/Q)

n2 = 1
n1 = 0
n0 = wz**2

k = (n2 * (wz/w0)**2) / (1 - Q0/Q)
m = k * ((K-1)/K) * (1 + 2*Q0**2 * (w0/wz)**2)

# Paso 2
C21 = 1e-9
C22_ideal = C21*(m/(1-m))
# Se elige a mano para que el cociente C22/C21 sea mas peque y de paso C22 sea nominal
#C22 = inferior_nominal_value(C22_ideal)
C22 = 12E-9

C3_ideal = C21 + C22
C3 = C3_ideal
#C3 = nearest_nominal_value(C3_ideal)     # C3 es aprox. la suma de arriba

# Paso 3
wp = w0 * (1 + Q0*w0/wt)
Qp = Q*(1 - 2*Q0*Q * (w0/wt) * (1/(2*Q)-w0/wt))

# Paso 4
G1 = 2 * Q0 * wp * np.sqrt(C3*(C21+C22))

G41pG42 = G1 / (4*Q0**2)  # G41 + G42

# Paso 5
Gb = 1/(1E3)

Ga1pGa2 = Gb * ((G41pG42/G1) * (C21 + C22 + C3)/C3 -
                wp*(C21 + C22)/(Qp*G1))    # Ga1 + Ga2

Ga2, G42 = sp.symbols('Ga2, G42')

# Paso 6
eq1 = sp.Eq(n1, (Ga1pGa2+Gb)/Gb * G42 * (1/(C21 + C22) + 1/C3) -
            Ga2/Gb * (G1/(C21 + C22) + G41pG42*(1/(C21 + C22) + 1/C3)))

# Paso 7
eqn2 = (Ga1pGa2+Gb)/Gb * (C22/(C21+C22)) - Ga2/Gb
eqn0 = G1/C3 * G41pG42/(C21 + C22) * ((G42/G41pG42) * (Ga1pGa2+Gb)/Gb - Ga2/Gb)

eq2 = sp.Eq(wz**2, eqn0/eqn2)

sol = sp.solve([eq1, eq2], [G42, Ga2], dict=True)

G42 = sol[0][G42]
Ga2 = sol[0][Ga2]

G41 = G41pG42 - G42
Ga1 = Ga1pGa2 - Ga2

n2 = (Ga1pGa2+Gb)/Gb * (C22/(C21+C22)) - Ga2/Gb #Real value of the gain constant

print('A continuación los valores finales que hay que poner en el circuito:')

print('R1 = ', 1 / (G1))
print('Ra1 = ', 1 / (Ga1))
print('Ra2 = ', 1 / (Ga2))
print('Rb = ', 1 / (Gb))
print('R41 = ', 1 / (G41))
print('R42 = ', 1 / (G42))
print('C3 = ', C3)
print('C22 = ', C22)
print('C21 = ', C21)

print('\n\n Para el RC:')
print('R = ', R)
print('C = ', C)


## Ahora tocan los cálculos de las sensibilidades