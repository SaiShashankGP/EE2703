"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 9
Date        : 11-05-2022
Description : Spectra of Non Periodic Signals
Language    : Python3

"""

# Import useful libraries
from pylab import *
import numpy as np

# Defining some important signals' classes so that they come in handy
class sinsq2t:
    def __init__(self, t):
        self.t = t
        self.name = r'$\sin({\sqrt{2}}t)$'
    def calc(self):
        return np.sin(np.sqrt(2)*self.t)

class Hamming:
    def __init__(self, N):
        self.N = N
    def calc(self):
        n = np.arange(self.N)
        return 0.54 + 0.46*np.cos((2*np.pi*n)/(self.N -1))

class sinwt:
    def __init__(self, t):
        self.t = t
        self.name = r'$\sin({1.25}t)$'
    def calc(self):
        return np.sin(1.25*self.t)

class cos3wt:
    def __init__(self, t):
        self.t = t
        self.w = 0.86
        self.name = rf'$\cos^3({self.w}t)$'
    def calc(self):
        return np.cos(self.w * self.t)**3

class cosine:
    def __init__(self, t):
        self.t = t
        self.w = 1.2
        self.d = 0.5
        self.name = r'$\cos({{\omega}_0}+{\delta})$'
    def calc(self):
        return np.cos(self.w * self.t + self.d)

class chirp:
    def __init__(self, t):
        self.t = t
        self.name = r'$\cos({16 \left({1.5} + {\frac{t}{2\pi}}\right)t})$'
    def calc(self):
        var = 16*(1.5 + (self.t/(2*np.pi)))*self.t
        return np.cos(var)

# Defining helper function
def dft(func, lim=10, start=-np.pi, end=np.pi, freq=64, plotGreen=False, Ham=False, Noise=False, Show=True):
    global x, fmax
    x = np.linspace(start, end, freq+1)[:-1]
    fmax = 1/(x[1]-x[0])
    y_ = func(x)
    if func == cosine:
        global w_act, d_act 
        w_act = y_.w
        d_act = y_.d
    y_name = y_.name
    y = y_.calc()
    if Noise:
        y += 0.1*randn(freq)
    if Ham:
        h_ = Hamming(freq)
        h = fftshift(h_.calc())
        y = y*h
        y_name = y_.name + r'$\times w(t)$'
    y = fftshift(y)
    global Y, w 
    Y = fftshift(fft(y))/freq
    w = np.linspace(-np.pi*fmax, np.pi*fmax, freq+1)[:-1]
    if Show:
        figure()
        subplot(2,1,1)
        plot(w,abs(Y))
        xlim([-1*lim,lim])
        ylabel(r"$|Y|$",size=16)
        title(rf"Spectrum of {y_name}")
        grid(True)
        subplot(2,1,2)
        plot(w,angle(Y),'ro')
        ii=where(abs(Y)>1e-3)
        plot(w[ii],angle(Y[ii]),'go',lw=2, visible=plotGreen)
        xlim([-1*lim,lim])
        ylabel(r"Phase of $Y$",size=16)
        xlabel(r"$k$",size=16)
        grid(True)
        show()
    

def weightedavg(samples, weights):
    num = np.dot(samples, weights)
    den = np.sum(weights)
    return num/den

# Q1: sin(sqrt(2)*t)
def Q1():
    '''Solves Q1'''
    # Normal DFT of sinsq2t
    dft(func=sinsq2t)    
    # DFT using Hamming's window
    dft(func=sinsq2t, Ham=True, lim=8)    
    # DFT using Hamming's window and more samples
    dft(func=sinsq2t, Ham=True, freq=256, start=-4*np.pi, end=4*np.pi, lim=4)

# Q2: cos^3(w*t) where w = 0.86
def Q2():
    '''Solves Q2'''
    # Without Hamming window
    dft(func=cos3wt, start=-4*np.pi, end=4*np.pi, freq=256, Ham=False)
    # With Hamming window
    dft(func=cos3wt, start=-4*np.pi, end=4*np.pi, freq=256, Ham=True)

# Q3: Estimate w, d for cos(wt+d)
def Q3():
    '''Solves Q3'''
    dft(func=cosine, start=-np.pi, end=np.pi, freq=128, lim=4, Ham=True)
    # w_0 is to be estimated as the weighted average of w's with weights as |Y(jw)|^2
    ii = np.where(w>=0)
    w_calc = weightedavg(samples=w[ii], weights=np.abs(Y[ii])**2)
    # Finding the nearest value to w_calc in w array and finding the phase of Y(jw) at that w
    jj = np.argmin(np.abs(w-w_calc))
    d_calc = np.angle(Y[jj])
    print(f'Actual w used\t: {w_act}')
    print(f'Estd. w\t: {w_calc.round(3)}')
    print(f'Actual d used\t: {d_act}')
    print(f'Estd. d\t: {d_calc.round(3)}')

# Q4: Estimate w, d for cos(wt+d) + white noise
def Q4():
    '''Solves Q4'''
    dft(func=cosine, start=-np.pi, end=np.pi, freq=128, lim=4, Noise=True, Ham=True)
    # w_0 is to be estimated as the weighted average of w's with weights as |Y(jw)|^2
    ii = np.where(w>=0)
    w_calc = weightedavg(samples=w[ii], weights=np.abs(Y[ii])**2)
    # Finding the nearest value to w_calc in w array and finding the phase of Y(jw) at that w
    jj = np.argmin(np.abs(w-w_calc))
    d_calc = np.angle(Y[jj])
    print(f'Actual w used\t: {w_act}')
    print(f'Estd. w\t: {w_calc.round(3)}')
    print(f'Actual d used\t: {d_act}')
    print(f'Estd. d\t: {d_calc.round(3)}')

# Q5: Analysis of chirped signal
def Q5():
    '''Solves Q5'''
    # Chirped signal without windowing
    dft(func=chirp, start=-np.pi, end=np.pi, freq=1024, lim=100, Ham=False)
    # Chirped signal with windowing
    dft(func=chirp, start=-np.pi, end=np.pi, freq=1024, lim=100, Ham=True)

# Q6: Analysis of chirped signal's time - frequency behaviour
def Q6():
    '''Solves Q6'''
    dft(func=chirp, start=-np.pi, end=np.pi, freq=1024, lim=100, Ham=True, Show=False)
    t = x[::64]
    t_ = np.reshape(x, (16, 64))
    Ymag = np.zeros((16, 64))
    Yphase = np.zeros((16, 64))
    for i in range(16):
        dft(func=chirp, start=t_[i][0], end=t_[i][-1], Ham=True, freq=64, Show=False)
        Ymag[i] = np.abs(Y)
        Yphase[i] = np.angle(Y)
    w = linspace(-fmax*np.pi,fmax*np.pi,65)
    w = w[:-1]
    t,w = meshgrid(t,w)
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(w,t,Ymag.T,cmap='viridis',linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('surface plot')
    ylabel(r"$\omega\rightarrow$")
    xlabel(r"$t\rightarrow$")
    show()

