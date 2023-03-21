# module load python/3.6.1
# export PYTHONPATH=/home/lbonnet/THESE/module/lib/python3.6/site-packages

import numpy as np
from .JacobiGQ import JacobiGQ
from .JacobiGL import JacobiGL
from .JacobiGR import JacobiGR
from scipy.special import gamma, factorial
def JNodeWt(NN,IP,alpha,beta):
# Compute interpolation points from Jacobi polynomials
# NN   is the order of interpolation (Nombre de poids-1)
# NN+1 is the number of points
# ES 14/10/2009 after Hesthaven, Gottlieb, Gottlieb (2007) and F.L. Gozman (2005)
    N1 = NN+1
    N2 = 2.*NN

    if (alpha==-0.5 and beta==alpha): # Chebyshev quadratures using exact formulae
        IJ = np.arange(0,NN+1)[:, np.newaxis]
        if IP == 1: # Gauss quadrature (exact for orders up to 2*NN+1)
            Xinter = -np.cos((2*IJ+1)*np.pi/(N2+2))
            Winter = (np.pi/N1)*np.ones( (N1,1) )
        if IP == 2: # Gauss-Lobatto quadrature (exact for orders up to 2*NN-1)
            Xinter = -np.cos(np.pi*IJ/NN)
            Winter = np.pi/NN*[[0.5], [np.ones((NN-1,1))], [0.5]]
        if IP == 3: # Gauss-Radau quadrature with X_0 =-1 (exact for orders up to 2*NN)
            Xinter = -np.cos(2.*np.pi*IJ/(N2+1))
            Winter = 2*np.np.pi/(N2+1)*[[0.5], [np.ones((NN,1))]]
        if IP == 4: # Gauss-Radau quadrature with X_N1=+1 (exact for orders up to 2*NN)
            Xinter = np.flipud(np.cos(2.*np.pi*IJ/(N2+1)))
            Winter = 2*np.pi/(N2+1)*[[np.ones((NN,1))], [0.5]]
        if IP == 5: # Equidistant
            Xinter = 2*np.pi*IJ/N1
            Winter = 2*np.pi*np.ones((N1,1))/N1
        else:
            print('JNodeWt: option non developpee')

    else:
        if IP == 1: # Gauss quadrature (exact for orders up to 2*NN+1)
            [Xinter,Winter] = JacobiGQ(alpha,beta,NN)
        if IP == 2: # Gauss-Lobatto quadrature (exact for orders up to 2*NN-1)
            [Xinter,Winter] = JacobiGL(alpha,beta,NN)
        if IP == 3: # Gauss-Radau quadrature with X_0 =-1 (exact for orders up to 2*NN)
            [Xinter,Winter] = JacobiGR(alpha,beta,NN,'L')
        if IP == 4: # Gauss-Radau quadrature with X_N1=+1 (exact for orders up to 2*NN)
            [Xinter,Winter] = JacobiGR(alpha,beta,NN,'R')
        if IP == 5: # Equidistant
            Xinter = np.transpose(np.linspace(-1,1,N1))
            Winter = 2*np.ones((N1,1))/N1
    #  else:
    #      print('JNodeWt: option non developpee')

    # Zero-th moment of Gauss-Jacobi weight w(x) = (1-x)^alpha*(1+x)^beta
    mu0 = 2**(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+2)
    # print(mu0)
    # print(np.abs(np.sum(Winter)))
    if (np.abs(np.sum(Winter) - mu0) > 1.0E-10):
        # print('JNodeWt: sum of the weights %f is not zero-th moment mu0 = %f.\n',np.sum(Winter),mu0)
        print('error')
    return [Xinter,Winter]

# if __name__ == '__main__':
#     h = JNodeWt(9,2,0,0)
#     print(h)
