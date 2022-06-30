"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 3
Date        : 17-02-2022
Description : Fitting models into the given data using scientific python modules and visualisation modules
Language    : Python3

"""

# Importing useful modules

from pylab import *
from scipy.linalg import lstsq
import scipy.special as sp
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# Defining some constants
global N, k
N = 101
k = 9

# Q1: Create the data required to continue with and store it in fitting.dat file
def Q1():
    '''This function executes the task in Q1'''

    t=linspace(0,10,N)              
    y=1.05*sp.jn(2,t)-0.105*t       
    Y=meshgrid(y,ones(k),indexing='ij')[0]
    global scl
    scl=logspace(-1,-3,k)           
    n=dot(randn(N,k),diag(scl))     
    yy=Y+n

    savetxt("fitting.dat",c_[t,yy]) 
    print('Data file is created. Do you want to load it into the program? Type Q2')
    return

# Q2: Extract the data from the fitting.dat file
def Q2():
    '''This function executes the task in Q2'''
    global fileArray, time, data
    try:
        fileArray = np.loadtxt('fitting.dat')
    except FileNotFoundError:
        print('ERROR: File not created. Type Q1')
    time, data = fileArray[:, 0], fileArray[:, 1:]
    return

# Q3: Plot the noisy curves with labels as sigma in logspace values used
def Q3():
    '''This function executes the task in Q3'''
    Q2()
    Legend = list(np.round_(scl, decimals=3))

    figure(0)
    plot(time, data)
    xlabel(r'$t$', size=20)
    ylabel(r'$f(t) + n$', size=20)
    title(r'Figure 0')
    legend(Legend)
    show()

# Q4: Create a function g(t, A, B) which returns A*J2(t) + Bt and plot the true value along with the Q3 plot
def g(t, A, B):
    '''This function returns a numpy.1darray that resembles A*J2(t) + Bt when plotted against t'''
    return A*sp.jn(2, t) + B*t

def Q4():
    '''This function executes the task in Q4'''
    Q2()
    Legend = list(np.round_(scl, decimals=3))
    Legend.append('true value')
    data_ = np.hstack((data, np.reshape(np.transpose(g(time, 1.05, -0.105)), (N, 1))))

    plot(time, data_)
    xlabel(r'$t$', size=20)
    ylabel(r'$f(t) + n$', size=20)
    title(r'Q4: Data to be fitted to theory')
    legend(Legend, loc=1)
    show()

# Q5: Plot the error bars for the curves
def Q5():
    '''This function executes the task Q5'''
    Q2()
    plot(time, g(time, 1.05, -0.105))
    errorbar(time[::5], data[:, 0][::5], scl[0], fmt='ro')
    grid(True)
    legend(['f(t)', 'errorbar'])
    title(r'Q5: Data points for stdev=0.1 along with true function')
    show()

# Q6: Construct M & p matrices which when multiplied gives the function g(t, A, B) always
def Q6():
    '''This function validates the statement that matrix (J(t) t)*(A; B) = g(t, A, B)'''
    Q2()
    global M, p, g_
    list_M = [[sp.jn(2, i), i] for i in time]
    M = np.array(list_M)
    p = np.array(np.reshape(np.array([1.05, -0.105]), (2, )))
    g_ = np.dot(M, p)
    return np.array(g_ == g(time, 1.05, -0.105)).all()

# Q7: create an e_ij matrix where e_ij is MSE of the function with Ai, Bj as values where Ai is from [0, 2, 0.1] Bj is from  [-0.2, 0, 0.01]
def Q7():
    '''This function executes the task in Q7'''
    Q2()
    global E, A, B
    A = list(np.linspace(0, 2, 21))
    B = list(np.linspace(-0.2, 0, 21))
    E = np.zeros((21, 21))
    for i in range(len(A)):
        for j in range(len(B)):
            E[i][j] = mse(g(time, A[i], B[j]), data[:, 0])

# Q8: Create a contour plot using e matrix
def Q8():
    '''This function executes the task in Q8'''
    Q7()
    plot(1.05, -0.105, 'ro')
    annotate('Exact Value', (1.05, -0.105))
    contour(np.array(A), np.array(B), E)
    xlabel('A')
    ylabel('B')
    title(r'Q8: Contour plot of e_ij')
    show()

# Q9: Compute optimum A & B for data with sigma = 0.1 using scipy.linalg.lstsq
def Q9():
    '''This function executes the task in Q9'''
    _ = Q6()
    ABarr, *l = lstsq(M, g_)
    print(ABarr)

# Q10: Compute the A & B estimates for all the columns and plot the error variation 
def Q10():
    '''This function executes the task in Q10'''
    _ = Q6()
    Aerr_ = []
    Berr_ = []
    MSE = []
    for i in range(k):
        a, mse, *b= lstsq(M, data[:, i])
        Aerr_.append(abs(1.05-a[0]))
        Berr_.append(abs(-0.105-a[1]))
        MSE.append(mse)
    Aerr = np.array(Aerr_)
    Berr = np.array(Berr_)
    MSE = np.array(MSE)
    plot(scl, Aerr, 'ro--')
    plot(scl, Berr, 'bo--')
    plot(scl, MSE)
    legend(['Aerr', 'Berr', 'LMSE'])
    grid(True)
    title(r'Q10: Variation of error with noise')
    show()

# Q11: Replot all the graphs in loglog space
def Q11():
    '''This function executes the task in Q11'''
    _ = Q6()
    Aerr_ = []
    Berr_ = []
    MSE = []
    for i in range(k):
        M_ = np.c_[data[:, i], time]
        a, mse, *b= lstsq(M_, g_)
        Aerr_.append(abs(1.05-a[0]))
        Berr_.append(abs(-0.105-a[1]))
        MSE.append(mse)
    Aerr = np.array(Aerr_)
    Berr = np.array(Berr_)
    MSE = np.array(MSE)
    loglog(scl, Aerr, 'ro--')
    loglog(scl, Berr, 'bo--')
    loglog(scl, MSE)
    legend(['Aerr', 'Berr', 'LMSE'])
    grid(True)
    title(r'Q11: Variation of error with noise in loglog scale')
    show()

# I/O section

exitBool = True
Q1()
while exitBool:
    cmd = input("Command: ")
    if cmd.lower() == 'exit':
        exitBool = False
    elif cmd.lower() == 'q2':
        Q2()
        print('Data file is loaded. Type Q3 for plotting data')
    elif cmd.lower() == 'q3':
        Q3()
    elif cmd.lower() == 'q4':
        Q4()
    elif cmd.lower() == 'q5':
        Q5()
    elif cmd.lower() == 'q6':
        print(Q6())
    elif cmd.lower() == 'q7':
        Q7()
        print('MSE matrix has been created. Do you want to see the contour plot? Type Q8')
    elif cmd.lower() == 'q8':
        Q8()
    elif cmd.lower() == 'q9':
        Q9()
    elif cmd.lower() == 'q10':
        Q10()
    elif cmd.lower() == 'q11':
        Q11()



    