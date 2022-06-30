"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : ENDSEM
Date        : 12-05-2022
Description : Find Currents in Half Wave Dipole Antenna
Language    : Python3

"""

# import required libraries
from pylab import *
import numpy as np

# Define every given and known parameters
'''
half length, speed of light, permeability in vacuum, no.of divisions, input current amplitude, radius, frequency, wave number, min. step
'''
l = 0.5
c = 2.9979e8
mu0 = 4*np.pi*1e-7
N = 4
# N = 100
Im = 1.0
a = 0.01
lambda_ = l*4
f = c/lambda_
k = 2*np.pi/lambda_
dz = l/N

# Q1: Setting up currents
'''
I defined indices of divisions as [-N, N] and multiplied it with min.step to get each divisions's z-coordinate
Similarly, for the unknown currents, I seperately defined the unknown indices left and right of centre and joined them 
Actual current is calculated using numpy.sin 
'''
i = np.arange(-N, N+1, 1)   
z = i*dz
j1 = np.arange(-N+1, 0, 1)
j2 = np.arange(1, N, 1)
j = np.r_[j1, j2]
u = j*dz
I = np.zeros(2*N +1)
I[N] = Im
J = np.zeros(2*N -2)

I_actual = Im*np.sin(k*(l-np.abs(z)))

# Q2: Create M, H matrices
'''
I defined an identity matrix and multipled it with the scalar value of 1/2*pi*a to get M matrix
'''
def computeM(a, N):
    const = 1/(2*np.pi*a)
    M = const*np.identity(2*N -2, dtype=float)
    return M

M = computeM(a, N)


# Q3: Computing A(r,z)
'''
R is calculated using r and z-z' matrices. multiply z-z' matrix with 1j, add them and take absolute value
from R, we can calculate P and Pb from the formual given in the question
'''
rz = a*np.ones(2*N +1, dtype=float)
ru = a*np.ones(2*N -2, dtype=float)
zz = z
zu = u
def findR(r, z, Z):
    z_comp1, z_comp2 = np.meshgrid(z,Z)
    z_comp = 1j*(z_comp1 - z_comp2)
    r_comp = np.meshgrid(r, r)[0]
    R = r_comp + z_comp
    R_val = np.abs(R)
    return R_val

Rz = findR(rz, zz, z)
Ru = findR(ru, zu, u)


def calcP(R):
    const = mu0/(4*np.pi)
    P = const*np.exp(-1j*k*R)*dz/R
    return P
def calcPB(Rn):
    const = mu0/(4*np.pi)
    Rn = list(Rn)
    Rn.pop(N)
    R = np.array(Rn[1:-1])
    P = -1*const*np.exp(-1j*k*R)*dz/R
    return P

P = calcP(Ru)
PB = calcPB(Rz[N])



# Computing H using A
'''
After obtaining P and Pb, we can calcualte Q, Qb can be caluculated with given formula in question
'''
def calcQ(P, R, r):
    const1 = 1/mu0
    const2 = (-1/R**2) + 1j*(-k/R)
    return -1*P*r*const1*const2

def calcQB(PB, Rn, r):
    const1 = 1/mu0
    Rn = list(Rn)
    Rn.pop(N)
    R = np.array(Rn[1:-1])
    const2 = (-1/R**2) + 1j*(-k/R)
    return -1*PB*r*const1*const2

Q = calcQ(P, Ru, np.meshgrid(ru, ru)[0])
QB = calcQB(PB, Rz[N], rz[N]*np.ones(2*N -2))



# Q5: Calculate the unknown currents J
'''
Using inv() and numpy.matmul(), we can find J. After finding J, 
I converted it to list and using insert() function, I added i=0, i=2*N, i=N currents
Final currents in the Antenna will be the real components of the current obtained. 
Taking the absolute value will neglect the drection's effect
'''
A = inv(M-Q)
B = QB*Im
J = np.matmul(A, B)
J = list(J)
J.insert(0, 0)
J.insert(2*N-1, 0)
J.insert(N, Im)
I_calc = np.abs((np.array(J)))

#Plot the currents with z-coordinates
plot(z, I_calc, 'r-', label=r'$I_{calc}$')
plot(z, I_actual, 'b-', label=r'$I_{model}$')
title(f'Actual v/s Calculated currents')
legend()
xlabel(r'z$\longrightarrow$')
ylabel(r'I$\longrightarrow$')
show()

if N == 4:
    print(f'z: {z.round(2)}')
    print('\n')
    print(f'u: {u.round(2)}')
    print('\n')
    print(f'Rz: {Rz.round(2)}')
    print('\n')
    print(f'Ru: {Ru.round(2)}')
    print('\n')
    print(f'P: {(P*1e8).round(2)}')
    print('\n')
    print(f'Pb: {(PB*1e8).round(2)}')
    print('\n')
    print(f'Q: {Q.round(2)}')
    print('\n')
    print(f'Qb: {QB.round(2)}')
    print('\n')
    print(f'Currents: {I}')