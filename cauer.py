from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

order, wn = signal.ellipord(2*np.pi*13e3, 2*np.pi*13e3/4, 2, 40, analog=True)
z, p, k = signal.ellip(order, 1.8, 45, wn,analog=True, output='zpk', btype='highpass')
# z, p, k = signal.lp2hp_zpk(z, p, k, 1)
print(order)
sos = signal.zpk2sos(z, p, k)
b, a = signal.zpk2tf(z, p, k)
tf = signal.TransferFunction(b, a)
wx = np.logspace(2, 5, 5000, True, 10)
w, m , ph = signal.bode(tf, w=wx)
plt.semilogx(w/2/np.pi, m)
print(sos)
print(k)
#plt.show()
fig, ax = plt.subplots()
ax.scatter(np.real(p), np.imag(p), marker='x', color='Blue')
ax.scatter(np.real(z), np.imag(z), marker='o', color='Red')
#plt.show()


K,q,q0,ga,gb,g1,g,C,w0,g4 = sp.symbols('K,q,q0,ga,gb,g1,g,C,w0,g4')
eq0=sp.Eq(K, 1+1/(2*q0**2)*(1-q0/q))
eq1 = sp.Eq(ga,(K-1)*gb)
eq2 = sp.Eq(g1,4*q0**2*g)
eq3 = sp.Eq(C,g*w0/(2*q0))
eq4 = sp.Eq(g4,g)

eqs = [eq0,eq1,eq2,eq3,eq4]

dic1 = sp.solve(eqs, [K,ga,g1,C,g4]) 


#print(dic1)
K = dic1[K]
ga = dic1[ga]
g1 = dic1[g1]
C = dic1[C]
g4 = dic1[g4]


K = K.subs(q0, 2.5)
K= K.subs(q, 2.63)
ga = ga.subs(gb, 1E-3)
ga = ga.subs(q0, 2.5)
ga = ga.subs(q, 2.63)
g1 = g1.subs(q0,2.5)
g1 = g1.subs(g, 0.1E-6)
C = C.subs(q0,2.5)
C = C.subs(g, 0.1E-6)
C = C.subs(w0, 85.5E3)
g4 = g4.subs(g, 0.1E-6)



print("K = ",K)
print("Ga = ",ga)
print ("G1 = ",g1)
print ("G4 = ",g)
print ("C2 = C3 = C = ",C)




