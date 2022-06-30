"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 4
Date        : 24-02-2022
Description : Finding Fourier approximations of two functions
Language    : Python3

"""

# Importing useful modules

from pylab import *
import numpy as np
from scipy.integrate import quad
from scipy.linalg import lstsq

# Defining some useful functions

def exp(x):
    '''This function returns e^x'''
    return np.exp(x)

def ccx(x):
    '''This function returns cos(cos(x))'''
    return np.cos(np.cos(x))

def u(x, f, k):
    '''This function returns f(x)cos(kx) for given arguments'''
    return f(x)*np.cos(k*x)

def v(x, f, k):
    '''This function returns f(x)sin(kx) for given arguments'''
    return f(x)*np.sin(k*x)

def findCoeffs(f, n_iter):
    '''This function finds the fourier coefficients of a given function f(x) where n = n_iter'''
    na = n_iter//2 + 1
    nb = n_iter//2
    f_a = np.zeros(na)
    f_b = np.zeros(nb)
    for i in range(1, na):
        f_a[i] = 1/np.pi * quad(u, 0, 2*np.pi, args=(f, i))[0]
    for i in range(nb):
        f_b[i] = 1/np.pi * quad(v, 0, 2*np.pi, args=(f, i+1))[0]
    f_a[0] = 1/(2*np.pi) * quad(u, 0, 2*np.pi, args=(f, 0))[0]
    return f_a, f_b

def constructCoeffVector(a_arr, b_arr):
    '''This function takes in a, b arrays and construct the actual coefficient sequence as a vector'''
    aList = list(a_arr)
    bList = list(b_arr)
    finalList = [aList[0]]
    aList.pop(0)
    i = 0
    for j in range(2*len(bList)):
        if j % 2 == 0:
            finalList.append(aList[i])
        else:
            finalList.append(bList[i])
            i += 1
    return np.array(finalList)

def splitCoeffVector(arr):
    '''This function takes in actual coeff vector and splits it into a, b vectors'''
    aList = [arr[0]]
    arrList = list(arr)
    arrList.pop(0)
    aList += arrList[::2]
    bList = arrList[1::2]
    return np.array(aList), np.array(bList)

# Q1: Create a function that takes in a scalar/vector and returns the scalar/vector with the corrsponding f(x) values in it

def Q1():
    '''This function executes Q1'''
    x = np.array(np.arange(-2*np.pi, 4*np.pi, 0.01))
    ccx_arr = ccx(x)
    exp_arr = exp(x)
    plot(x, ccx_arr)
    legend(['cos(cos(x))'], loc='upper right')
    grid(True)
    title('Plot of cos(cos(x))')
    show()
    semilogy(x, exp_arr)
    legend(['exp(x)'])
    grid(True)
    title('SemiLog Plot of exp(x)')
    show()

# Q2: Get the first 51 coefficients in the fourier series approximation of those two functions

n_iter = 51
global x1, x2
x1 = np.arange(0, n_iter//2 +1, 1)
x2 = np.arange(1, n_iter//2+1, 1)

def Q2():
    '''This function executes Q2'''
    global e_a, e_b, c_a, c_b
    e_a, e_b = findCoeffs(exp, n_iter)
    c_a, c_b = findCoeffs(ccx, n_iter)

# Q3: Plot coefficients vs n for each function

def Q3():
    '''This function executes Q3'''
    Q2()
    figure(3)
    semilogy(x1, np.abs(e_a), 'ro')
    semilogy(x2, np.abs(e_b), 'ro')
    grid(True)
    xlabel('n')
    ylabel(r'$a_n, b_n$')
    title('SemiLog Plot of coeffs of exp(x)')
    show()
    figure(4)
    loglog(x1, np.abs(e_a), 'ro')
    loglog(x2, np.abs(e_b), 'ro')
    grid(True)
    xlabel('n')
    ylabel(r'$a_n, b_n$')
    title('LogLog plot of coeffs of exp(x)')
    show()
    figure(5)
    semilogy(x1, np.abs(c_a), 'ro')
    semilogy(x2, np.abs(c_b), 'ro')
    grid(True)
    xlabel('n')
    ylabel(r'$a_n, b_n$')
    title('SemilLog Plot of coeffs of ccx(x)')
    show()
    figure(6)
    loglog(x1, np.abs(c_a), 'ro')
    loglog(x2, np.abs(c_b), 'ro')
    grid(True)
    xlabel('n')
    ylabel(r'$a_n, b_n$')
    title('LogLog Plot of coeffs of ccx(x)')
    show()

# Q4: Find coefficients from matrix multiplication

global x
x = np.linspace(0, 2*np.pi, 401)[:-1]
be = exp(x)
bc = ccx(x)
global A, ce, cc
A = np.zeros((400, n_iter))
A[:, 0] = 1
for k in range(1, n_iter//2 +1):
    A[:, 2*k-1] = np.cos(k*x)
    A[:, 2*k] = np.sin(k*x)
ce = lstsq(A, be)[0]
cc = lstsq(A, bc)[0]
cc_a, cc_b = splitCoeffVector(cc)
ce_a, ce_b = splitCoeffVector(ce)

def Q4():
    '''This function executes Q4&5'''
    Q2()
    semilogy(x1, np.abs(e_a), 'ro', label='Actual coeffs')
    semilogy(x2, np.abs(e_b), 'ro')
    semilogy(x1, np.abs(ce_a), 'go', label='Calculated coeffs')
    semilogy(x2, np.abs(ce_b), 'go')
    title('SemiLog Plot of actual & calculated coeffs of exp(x)')
    grid(True)
    legend()
    show()
    semilogy(x1, np.abs(c_a), 'ro', label='Actual coeffs')
    semilogy(x2, np.abs(c_b), 'ro')
    semilogy(x1, np.abs(cc_a), 'go', label='Calculated coeffs')
    semilogy(x2, np.abs(cc_b), 'go')
    title('SemiLog Plot of actual & calculated coeffs of ccx(x)')
    grid(True)
    legend()
    show()

# Q6: Find the deviation(maximum abs. difference between coeffs by integration & lstsq) 

def Q6():
    '''This function executes Q6'''
    ce_actual = constructCoeffVector(e_a, e_b)
    cc_actual = constructCoeffVector(c_a, c_b)
    maxdev_exp = np.max(np.abs(ce-ce_actual))
    maxdev_ccx = np.max(np.abs(cc-cc_actual))
    print(f'Max. deviation in coeffs of exp(x): {maxdev_exp}')
    print(f'Max. deviation in coeffs of ccx(x): {maxdev_ccx}')

# Q7: Calculate the functions exp(x) and ccx(x) using the obtained coeffs from lstsq

def Q7():
    '''This function executes Q7'''
    calc_exp = np.matmul(A, ce)
    calc_ccx = np.matmul(A, cc)
    semilogy(x, exp(x), 'ro', label='Actual Function')
    semilogy(x, calc_exp, 'go', label='Calulated Function')
    legend(loc='lower right')
    grid(True)
    title('Actual vs Calculated of exp(x)')
    show()
    plot(x, ccx(x), 'ro', label='Actual Function')
    plot(x, calc_ccx, 'go', label='Calulated Function', markersize=2)
    grid(True)
    legend(loc='lower right')
    title('Actual vs Calculated of ccx(x)')
    show()

exitBool = True

while exitBool:
    cmd = input('Command: ')
    if cmd.lower() == 'exit':
        exitBool = False
    if cmd.lower() == 'q1':
        Q1()
    if cmd.lower() == 'q2':
        Q2()
    if cmd.lower() == 'q3':
        Q3()
    if cmd.lower() == 'q4':
        Q4()
    if cmd.lower() == 'q6':
        Q6()
    if cmd.lower() == 'q7':
        Q7()



