"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 8
Date        : 28-03-2022
Description : DTFT and DFT using Numpy
Language    : Python3

"""

# Import required libraries
from pylab import *
import sys

# Defining some important functions and classes that might come handy
class sin5x:
    def __init__(self, t):
        self.t = t
        self.name = r'$\sin(5x)$'
    def calc(self):
        return np.sin(5*self.t)

class AMexample:
    def __init__(self, t):
        self.t = t
        self.name = r'$(1+0.1\cos(x))\cos(10x)$'
    def calc(self):
        return (1 + 0.1*np.cos(self.t))*np.cos(10*self.t)

class sinx3:
    def __init__(self, t):
        self.t = t
        self.name = r'${\sin ^3(x)}$'
    def calc(self):
        val = np.sin(self.t)
        return val**3

class cosx3:
    def __init__(self, t):
        self.t = t
        self.name = r'${\cos ^3(x)}$'
    def calc(self):
        val = np.cos(self.t)
        return val**3

class FMwave:
    def __init__(self, t):
        self.t = t
        self.name = r'$\cos(20t + 5\cos(t))$'
    def calc(self):
        return np.cos(20*self.t + 5*np.cos(self.t))

class Gaussian:
    def __init__(self, t):
        self.t = t
        self.name = r'$e^{\frac{-t^2}{2}}$'
    def calc(self):
        return np.exp(-0.5*self.t**2)
    def acc_ft(self, w):
        return np.sqrt(2*np.pi)*np.exp(-0.5*w**2)


    

def helper_(func, lim=10, start=0, end=2*np.pi, freq=128, plotRed=True):
    '''This function helps us automate the task of solving Q1'''
    x = np.linspace(start, end, freq+1)[:-1]
    y_=func(x)
    y_name = y_.name
    y = y_.calc()
    Y=fftshift(fft(y))/freq
    w=linspace(-64,64,freq+1)[:-1]
    if func == Gaussian:
        Y = Y/max(Y)
        max_err = max(Y-y_.acc_ft(w))
        print(max_err)
    figure()
    subplot(2,1,1)
    plot(w,abs(Y))
    xlim([-1*lim,lim])
    ylabel(r"$|Y|$",size=16)
    title(rf"Spectrum of {y_name}")
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'ro', visible=plotRed)
    ii=where(abs(Y)>1e-3)
    plot(w[ii],angle(Y[ii]),'go',lw=2)
    xlim([-1*lim,lim])
    # if func == Gaussian:
    #     ylim([-2, 2])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$k$",size=16)
    grid(True)
    show()

# Q1 Part a: sin(5t)
def Q1a():
    '''Solves Q1 for sin(5t)'''
    helper_(func=sin5x)

# Q1 Part b: (1+0.1cos(t))cos(10t)
def Q1b():
    '''Solves Q1 for (1+0.1cos(t))cos(10t)'''
    helper_(func=AMexample, lim=15, start=-4*np.pi, end=4*np.pi, freq=512)

# Q2 Part a: sin^3(t)
def Q2a():
    '''Solves Q2 for sin^3(t)'''
    helper_(func=sinx3, lim=6, start=-4*np.pi, end=4*np.pi, freq=512)

# Q2 Part b: cos^3(t)
def Q2b():
    '''Solves Q2 for cos^3(t)'''
    helper_(func=cosx3, lim=6, start=-4*np.pi, end=4*np.pi, freq=512)

# Q3: cos(20t+5cos(t)) or example of FM wave
def Q3():
    '''Solves Q3 for cos(20t+5cos(t))'''
    helper_(func=FMwave, lim=50, start=-128*np.pi, end=128*np.pi, freq=8192, plotRed=False)

# Q4: exp(-t^2/2)
def Q4():
    '''Solves Q4 for  exp(−t^2/2)'''
    helper_(func=Gaussian, lim=10, start=-8*np.pi, end=8*np.pi, freq=1024, plotRed=True)

Q4()