"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 7
Date        : 28-03-2022
Description : Circuit Analysis using Sympy
Language    : Python3

"""

# Importing useful libraries and modules
from pylab import *
import numpy as np
import scipy.signal as sp
import sympy as sm

# Defining some useful functions which might come in handy
def vi(t):
    '''Returns the input voltage V_i(t) in Q2'''
    return np.where(t<0, 0, np.sin(2e3*np.pi*t) + np.cos(2e6*np.pi*t)), np.where(t<0, 0, np.sin(2e3*np.pi*t))

def u(t):
    '''Returns the unit step function'''
    return np.where(t<0, 0, 1)

def ds(t, w:float=1.5, a:float=0.05):
    return np.where(t<0, 0, np.sin(w*t)*np.exp(-1*a*t))

def symExpToLTI(exp, s=sm.symbols('s')):
    '''Returns the scipy lti object'''
    exp_num, exp_den = exp.as_numer_denom()
    num_coeffs = sm.Poly(exp_num, s).all_coeffs()
    den_coeffs = sm.Poly(exp_den, s).all_coeffs()
    num = poly1d([float(n) for n in num_coeffs])
    den = poly1d([float(n) for n in den_coeffs])
    return sp.lti(num, den)

def LPF(Vi=1, R1:float=10e3, R2:float=10e3, C1:float=1e-9, C2:float=1e-9, G:float=1.586):
    '''Gives the Transfer function H(s) of given LPF in assignment'''

    s = sm.symbols('s')

    A = sm.Matrix([[(1/R1)+(1/R2)+(s*C1), 0, (-1/R2), (-s*C1)], [(-1/R2), 0, (1/R2)+(s*C2), 0], [0, G, 0, -1], [0, -G, G, -1]])
    B = sm.Matrix([Vi/R1, 0, 0, 0])

    V = A.inv()*B
    Vo = V[3]
    # print(Vo)

    ww = np.logspace(0, 10, 1001)
    ss = 1j*ww
    hf = sm.lambdify(s, Vo, 'numpy')
    v = hf(ss)

    loglog(ww, abs(v), lw=2)
    title(r'$\mid H(j{\omega}\mid$ of given LPF')
    xlabel(r'$\omega (log) \longrightarrow$')
    ylabel(r'${\mid H(j{\omega}\mid} (dB) \longrightarrow$')
    grid(True)
    show()

    H = symExpToLTI(Vo)

    return Vo

def HPF(Vi=1, R1:float=10e3, R2:float=10e3, C1:float=1e-9, C2:float=1e-9, G:float=1.586):
    '''Gives the Transfer function H(s) of given HPF in assignment'''

    s = sm.symbols('s')

    A = sm.Matrix([[0, G, 0, -1], [0, -G, G, -1], [(-s*C2), 0, (s*C2)+(1/R2), 0], [(s*(C1+C2)) + (1/R1), 0, (-s*C2), (-1/R1)]])
    B = sm.Matrix([0, 0, 0, Vi*s*C1])

    V = A.inv()*B
    global Vo
    Vo = V[3]
    # print(Vo)

    ww = np.logspace(0, 10, 1001)
    ss = 1j*ww
    hf = sm.lambdify(s, Vo, 'numpy')
    v = hf(ss)

    loglog(ww, abs(v), lw=2)
    title(r'$\mid H(j{\omega}\mid$ of given HPF')
    xlabel(r'$\omega (log) \longrightarrow$')
    ylabel(r'${\mid H(j{\omega}\mid} (dB) \longrightarrow$')
    grid(True)
    show()

    H = symExpToLTI(Vo)

    return H

def Q1(t=np.linspace(0, 1e-3, 1001)):
    '''This function solves Q1'''
    H = LPF()
    u_ = u(t)

    _, u_resp, _ = sp.lsim(H, u_, t)

    plot(t, u_resp)
    title('Step response of given LPF')
    xlabel(r'time $\longrightarrow$')
    show()

def Q2(t=np.linspace(0, 10e-3, 1000001)):
    '''This function solves Q2'''
    Vo = LPF()
    inp, lfc = vi(t)

    _, resp, _ = sp.lsim(Vo, inp, t)

    plot(t, resp, label='Vo')
    plot(t, lfc, label='LFC(Vi)')
    title('LPF on mixed frequencies')
    xlabel(r'time $\longrightarrow$')
    legend(loc='upper right')
    show()

def Q3():
    '''This function solves Q3'''
    H = HPF()

    print(Vo)

def Q4(t=np.linspace(1e-3, 30, 10001)):
    '''This function solves Q4'''
    H = HPF()
    inp= ds(t)

    _, resp, _ = sp.lsim(H, inp, t)

    plot(t, resp, label='Vo')
    plot(t, inp, label='Vi')
    title('HPF on a damped sinusoid with low frequency')
    xlabel(r'time $\longrightarrow$')
    legend(loc='upper right')
    show()

def Q5(t=np.linspace(1e-8, 1e-3, 10001)):
    '''This function solves Q5'''
    H = HPF()
    u_ = u(t)

    _, u_resp, _ = sp.lsim(H, u_, t)

    plot(t, u_resp)
    title('Step response of given HPF')
    xlabel(r'time $\longrightarrow$')
    show()

print(LPF())


