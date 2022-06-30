"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 6
Date        : 27-03-2022
Description : Laplace Transforms related questions and solving them in python
Language    : Python3

"""

# importing useful libraries
from pylab import *
import numpy as np
import scipy.signal as sp

# Defining the time interval
global N, t
N = 1000
t = np.linspace(0, 50, N)

# Defining the fuction f(t) and vi(t)
def f(t, a:float=0.5, w:float=1.5):
    return np.where(t<=0, 0, np.cos(w*t)*np.exp(-1*a*t))

def vi(t):
    return np.where(t <= 0, 0, np.cos(10**3 * t) - np.cos(10**6 * t))

def Q1(a:float=0.5, w:float=1.5, init_x:float=0.0, init_xdot:float=0.0):
    '''
    This function solves the question 1
    '''

    # Defining the laplace transform of function f(t) viz F
    global F_num,  F_den, F

    F_num = poly1d([1, a])
    F_den = poly1d([1, 2*a, w**2+a**2])

    F = sp.lti(F_num, F_den)

    # Defining the differential equation
    # x`` + 2.25x = f(t) in time domain
    # s^2 . X - s . x`(0) - x(0) - 2.25 . X = F in laplace domain

    global X_num, X_den, X

    X_num = polyadd(F_num, polymul(F_den, [init_x, init_xdot]))
    X_den = polymul(F_den, [1, 0, w**2])

    X = sp.lti(X_num, X_den)

    _, x = sp.impulse(X, None, t)

    # Plot the obtained x(t) from 0 to 50 seconds
    figure(0)
    plot(t, x)
    xlabel(r'time $\longrightarrow$')
    ylabel(r'x(t)$\longrightarrow $ ')
    title(r'solution of equation $\ddot x $ + 2.25x = f(t)')
    show()

def Q2():
    '''
    This function solves the question 2
    '''
    Q1(a=0.05)

def Q3():
    '''
    This function solves the question 3
    '''
    Q2()

    iterArr = np.arange(1.4, 1.6, 0.05)

    # Finding the transfer function X/F

    H_num = polymul(X_num, F_den)
    H_den = polymul(X_den, F_num)

    H = sp.lti(H_num, H_den)

    # Starting the loop and plotting

    t_ = np.linspace(0, 100, 2*N)

    for i in range(5):
        f_ = f(t_, a=0.05, w=iterArr[i])
        _, x, _ = sp.lsim(H, f_, t_)
        subplot(3, 2, i+1)
        plot(t_, x)

    show()

def Q4(init_x:float=1, init_y:float=0, init_xdot:float=0, init_ydot:float=0):
    '''
    This function solves question 4
    '''
    # Given, x`` + x - y = 0 and y`` + 2y - 2x = 0
    # Using this, x``(0) = y(0) - x(0)
    # y``(0) = 2x(0) - 2y(0)
    # x```(0) = y`(0) - x`(0)
    # Substituting eq1 in eq2, we get x```` + 3x`` = 0
    # In laplace, s^4 . K - s^3 . x(0) - s^2 . x`(0) - s . x``(0) - x```(0) + s^2 . K - s . x(0) - x`(0) = 0

    init_xddot = init_y - init_x
    init_xdddot = init_ydot - init_xdot

    K_num = poly1d([init_x, init_xdot, 3*init_x+init_xddot, 3*init_xdot+init_xdddot])
    K_dec = poly1d([1, 0, 3, 0, 0])

    L_num = polyadd(polymul(K_num, [1, 0, 1]), polymul(K_dec, [-1*init_x, 0]))
    L_dec = K_dec

    K = sp.lti(K_num, K_dec)
    L = sp.lti(L_num, L_dec)

    _, k = sp.impulse(K, None, t)
    _, l = sp.impulse(L, None, t)

    plot(np.linspace(0, 20, N), k)
    xlabel(r'time $\longrightarrow$')
    ylabel(r'x(t) $\longrightarrow$')
    title('x(t) v/s t')
    show()

    plot(np.linspace(0, 20, N), l)
    xlabel(r'time $\longrightarrow$')
    ylabel(r'y(t) $\longrightarrow$')
    title('y(t) v/s t')
    show()

def Q5(R:float=100, L:float=1e-6, C:float=1e-6):
    '''
    This function solves question 5
    '''
    # Transfer function is of the form 1/(1+sRC+s^2 LC)

    global T_num, T_dec, T

    T_num = poly1d([1])
    T_dec = poly1d([L*C, R*C, 1])

    T = sp.lti(T_num, T_dec)

    w, S, phi = T.bode()

    subplot(1, 2, 1)
    title('Magnitude Bode Plot of H(s)')
    xlabel('w(log)')
    ylabel(r'$\mid H(jw)\mid$ (log)')
    semilogx(w, S)
    subplot(1, 2, 2)
    title('Phase Bode Plot of H(s)')
    xlabel('w(log)')
    ylabel(r'$\angle H(jw)$ (log)')
    semilogx(w, phi)
    show()

def Q6():
    '''
    This function solves question 6
    '''
    Q5()
    t1 = np.arange(0, 30e-6, 1e-6)
    t2 = np.arange(0, 100e-3, 1e-6)

    _, vo1, _ = sp.lsim(T, vi(t1), t1)
    plot(t1, vo1)
    title(r'Output from 0 to 30$\mu$s')
    xlabel(r'time $\longrightarrow$')
    ylabel(r'${V_o}(t) \longrightarrow$')
    show()

    _, vo2, _ = sp.lsim(T, vi(t2), t2)
    plot(t2, vo2)
    title(r'Output from 0 to 100ms')
    xlabel(r'time $\longrightarrow$')
    ylabel(r'${V_o}(t) \longrightarrow$')
    show()
