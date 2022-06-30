"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 5
Date        : 08-03-2022
Description : Find the electric potential of each region in a resistive material which has potential difference across it
Language    : Python3

"""

# Importing useful libraries
import sys
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.linalg import lstsq

# Taking inputs for Nx, Ny, radius, Niter

Nx = int(sys.argv[1])       # Default should be 25
Ny =  int(sys.argv[2])       # Default should be 25
radius =  int(sys.argv[3])   # Default should be 8
Niter = int(sys.argv[4])    # Default should be 1500

# Initialising the potential array 
phi = np.zeros((Nx, Ny))

# Finding out what points lie on the 1V region
x = np.linspace(-1* Nx//2, Nx//2, Nx)
y = np.linspace(-1* Ny//2, Ny//2, Ny)

Y, X = np.meshgrid(y, x)

phi = np.where(X*X+Y*Y <= radius*radius, 1.0, 0.0)
ii = np.where(X*X + Y*Y <= 64)

# plot the contour plot of phi in initial iteration
figure(0)
plot(ii[0]-Nx//2, ii[1]-Ny//2, 'ro')
contour(Y, X, phi)
xlabel('x')
ylabel('y')
title('Initial Potential Distribution')
show()

# Similarly, with each iteration, we have to update phi matrix using the condition of laplace equation 
def updatePhi(phi):
    '''Updates phi matrix taking boundary conditions into consideration as well'''
    # For non-boundary points
    phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2] + phi[0:-2, 1:-1] + phi[1:-1, 2:] + phi[2:, 1:-1])
    # For boundary points
    phi[1:-1, 0] = phi[1:-1, 1]
    phi[1:-1, -1] = phi[1:-1, -2]
    phi[0, 1:-1] = phi[1, 1:-1]
    phi[-1, 1:-1] = 0
    # Correcting the potential at the contact points
    phi[ii] = 1
    # For corner points
    phi[0, 0] = 0.5*(phi[0, 1] + phi[1, 0])
    phi[0, -1] = 0.5*(phi[0, -2] + phi[1, -1])
    phi[-1, 0] = 0.5*(phi[-1, 1] + phi[-2, 0])
    phi[-1, -1] = 0.5*(phi[-1, -2] + phi[-2, -1])

errors = np.zeros(Niter)

for i in range(Niter):
    oldphi = np.copy(phi)
    updatePhi(phi)
    errors[i] = np.max(np.abs(oldphi-phi))

# Plot the error trend wrt iterations
figure(1)
semilogy(np.arange(0, Niter, 1)[::50], errors[::50], 'ro')
xlabel('No.of Iterations')
ylabel(r'${error}_i$')
title('Semilog Plot of Error v/s Iterations')
show()

figure(2)
loglog(np.arange(0, Niter, 1)[::50], errors[::50], 'ro')
xlabel('No.of Iterations')
ylabel(r'${error}_i$')
title('Loglog Plot of Error v/s Iterations')
show()

# It seems like error is exponentially varying with iteration count
# So, let's find out the best fit using scipy.linalg.lstsq
# loge = logA + BN

M1 = np.ones((Niter, 2))
M1[:, 1] = np.arange(0, Niter, 1)
b1 = np.log(errors)

coeffs_alliter, *f = lstsq(M1, b1)

M2 = np.ones((Niter-500, 2))
M2[:, 1] = np.arange(500, Niter, 1)
b2 = np.log(errors[500:])

coeffs_above500iter, *f = lstsq(M2, b2)

# print(coeffs_alliter, coeffs_above500iter)

def errorFit(x, logA, B):
    return np.exp(logA + B*x)

# Plot the two fits along with actual error plot 
figure(3)
semilogy(np.arange(0, Niter, 1)[::50], errors[::50], 'ro', label='actual')
semilogy(np.arange(0, Niter, 1)[::50], errorFit(np.arange(0, Niter, 1), coeffs_alliter[0], coeffs_alliter[1])[::50], 'bo', label='fit1')
xlabel('No.of Iterations')
ylabel(r'${error}_i$')
legend()
title('Semilog Plot of Fit no. 1 and Actual errors')
show()

figure(4)
semilogy(np.arange(0, Niter, 1)[::50], errors[::50], 'ro', label='actual')
semilogy(np.arange(0, Niter, 1)[::50], errorFit(np.arange(0, Niter, 1), coeffs_above500iter[0], coeffs_above500iter[1])[::50], 'go', label='fit2')
xlabel('No.of Iterations')
ylabel(r'${error}_i$')
legend()
title('Semilog Plot of Fit no. 2 and Actual errors')
show()

# Stopping condition: When cumulative error tends to stabilise, we can stop iterating. 
# But in our case, it doesn't stabilise. 
# It can be shown by considering the upper bound of the cumulative error always decreasing linearly in semilog plot

def cumulativeErrorFit(x, logA, B):
    return (-1/B)*np.exp(logA + B*0.5 + B*x)

figure(5)
semilogy(np.arange(0, Niter, 1)[::50], cumulativeErrorFit(np.arange(0, Niter, 1), coeffs_alliter[0], coeffs_alliter[1])[::50], 'ro')
xlabel('Cumulative Error')
ylabel('No.of iterations')
title('Cumulative Error v/s No.of iterations')
show()

# Surface plot of the potential after all the iterations
fig = figure(6)
ax = p3.Axes3D(fig)
surf = ax.plot_surface(Y, X,  phi.T, rstride=1, cstride=1, cmap=cm.jet)
xlabel('x in coordinates')
ylabel('y in coordinates')
ax.set_zlabel('Potential')
ax.set_title('Surface 3-D plot of the potential')
show()

# Contour Plot of the potential after all the iterations
figure(7)
contour(Y, X[::-1], phi)
plot(ii[0]-Nx//2-1, ii[1]-Ny//2-1, 'ro')
xlabel('x in coordinates')
ylabel('y in coordinates')
title('2-D Contour Plot of the potential')
show()

# Vector plots of the current
Jx = np.zeros((Nx, Ny))
Jy = np.zeros((Nx, Ny))
Jx[:, 1:-1] = -0.5*(phi[:, 2:] - phi[:, :-2])
Jy[1:-1, :] = 0.5*(phi[2:, :] - phi[:-2, :])
figure(8)
quiver(Y[2:-1], X[::-1][2:-1], Jx[2:-1], Jy[2:-1])
plot(ii[0]-Nx//2-1, ii[1]-Ny//2-1, 'ro')
title('Vector plot of the current flow')
show()




















