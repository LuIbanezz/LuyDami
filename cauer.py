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
#-----------------------------------------------------------------

K,q,q0,ga,gb,g1,g,C,w0,g4,g1,k,wz,n,m,h,c22,c21,wt,wp,qp,g41,g42,ga1,ga2 = sp.symbols('K,q,q0,ga,gb,g1,g,C,w0,g4,g1,k,wz,n,m,h,c22,c21,wt,wp,qp,g41,g42,ga1,ga2')
eq0=sp.Eq(K, 1+1/(2*q0**2)*(1-q0/q))
eq1 = sp.Eq(ga,(K-1)*gb)
#eq2 = sp.Eq(g1,4*q0**2*g)
eq3 = sp.Eq(C,g*w0/(2*q0))
eq4 = sp.Eq(g4,g)
eq5 = sp.Eq(k, ((wz/w0)**2/(1-q0/q)))
eq6 = sp.Eq(n, k*(1-q0/(K*q)))
eq7 = sp.Eq(m,k*((K-1)/K)*(1+2*q0**2*(w0/wz)**2))
eq8 = sp.Eq(h, m/(1-m))
eq9 = sp.Eq(c22/c21, h)
eq10 = sp.Eq(c22 + c21, C)
eq11 = sp.Eq(wp, w0*(1+q0*(w0/wt)))
eq12 = sp.Eq(g1, 2*q0*wp*(C*(c21+c22))**0.5)
eq13 = sp.Eq(qp, q*(1-2*q0*q*(w0/wt)*(1/(2*q)-w0/wt)))
eq14 = sp.Eq(g41+g42,g1/(4*q0**2))
eq15 = sp.Eq((ga1+ga2)/gb, ((g41+g42)/g1)*((c21+c22+C)/C)-(wp*(c21+c22)/(qp*g1)))
eq16 = sp.Eq(((ga1 + ga2 + gb)/(gb))*g42*((1/(c21+c22))+1/C)-(ga2/gb)*((g1/(c21+c22))+(g41+g42)*((1/(c21+c22))+1/C)), 0)
eq17 = sp.Eq(wz**2, (g1*(g41+g42))/(C*(c21+c22))*((g42/(g41+g42)*(ga1+ga2+gb)/gb)-(ga2/gb)))



eqs = [eq0,eq1,eq3,eq4,eq5,eq6,eq7,eq8, eq9, eq10,eq11,eq12,eq13,eq14,eq15,eq16]

dic1 = sp.solve(eqs, [K,ga,g1,g,g4,k,n,m,h, c22, c21,wp,qp,g41,g42,ga1,ga2]) 


#print(dic1)
K = dic1[K]
ga = dic1[ga]
g1 = dic1[g1]
g = dic1[g]
g4 = dic1[g4]
k = dic1[k]
n = dic1[n]
m = dic1[m]
h = dic1[h]
c22 = dic1[c22]
c21 = dic1[c21]
wp = dic1[wp]
qp = dic1[qp]
ga1 = dic1[ga1]
ga2 = dic1[ga2]
g41 = dic1[g41]
g42 = dic1[g42]

K = K.subs(q0, 1)
ga = ga.subs(q0, 1)
g1 = g1.subs(q0, 1)
g4 = g4.subs(q0, 1)
g = g.subs(q0, 1)
k = k.subs(q0, 1)
n = n.subs(q0, 1)
m = m.subs(q0, 1)
c21 = c21.subs(q0, 1)
c22 = c22.subs(q0, 1)
wp = wp.subs(q0, 1)
qp = qp.subs(q0, 1)

K= K.subs(q, 2.63)
ga = ga.subs(q, 2.63)
k = k.subs(q, 2.63)
n = n.subs(q, 2.63)
m = m.subs(q, 2.63)
c21 = c21.subs(q, 2.63)
c22 = c22.subs(q, 2.63)
qp = qp.subs(q, 2.63)


ga = ga.subs(gb, 1E-3)

g = g.subs(w0, 85.5E3)
k = k.subs(w0, 85.5E3)
n = n.subs(w0, 85.5E3)
m = m.subs(w0, 85.5E3)
c21= c21.subs(w0, 85.5E3)
c22 = c22.subs(w0, 85.5E3)
wp = wp.subs(w0, 85.5E3)
g1 = g1.subs(w0, 85.5E3)
qp = qp.subs(w0, 85.5E3)
g4 = g4.subs(w0, 85.5E3)

g1 = g1.subs(C, 100E-9)
g = g.subs(C, 100E-9)
g4 = g4.subs(C, 100E-9)
c21 = c21.subs(C, 100E-9)
c22 = c22.subs(C, 100E-9)

k = k.subs(wz, 27.53E3)
n = n.subs(wz, 27.53E3)
m = m.subs(wz, 27.53E3)
c21 = c21.subs(wz, 27.53E3)
c22 = c22.subs(wz, 27.53E3)

wp = wp.subs(wt, 17.17E6)
g1 = g1.subs(wt, 17.17E6)
qp = qp.subs(wt, 17.17E6)


print("K = ",K)
print("Ga = ",ga)
print ("G1 = ",g1)
print ("G4 = ",g4)
print ("C2 = C3 = C = ",100e-9)
print ("k = ", k)
print("n = ", n)
print("m = ", m)
print ( "c22/c21 = ", m/(1-m))
print ("c22 =", c22)
print("c21 = ",c21)
print("A0 = 204000, wp = 84.2")
print("wt =", 17.17E6)
print("wp =", wp)
print("qp =", qp)
print("ga1 =", ga1)
print("ga2 =", ga2)
print("g41 =", g41)
print("g42 =", g42)



