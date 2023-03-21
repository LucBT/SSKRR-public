from scipy.special import gamma, factorial
import numpy as np
from .JacobiGQ import JacobiGQ
from numpy import linalg as LA
def JacobiGR(alpha,beta,N,IR):
    # Evaluate N-th order Gauss-Radau quadrature points x and weights w
    # associated with Jacobi polynomials of type (alpha,beta) > -1
    # note: if N is the order, length(x) = N+1
    # ES 16/10/2009 after F.L. Gozman (2005)

    x = np.zeros((N+1,1))
    w = np.zeros((N+1,1))

    B0 = 2**(alpha+beta+1)*np.prod(np.arange(1,N+1))/gamma(alpha+beta+N+2)
    B1 = B0*gamma(beta +1)*gamma(beta +2)*gamma(alpha+N+1)/gamma(beta +N+2)
    B2 = B0*gamma(alpha+1)*gamma(alpha+2)*gamma(beta +N+1)/gamma(alpha+N+2)
    #
    if IR == 'L':
        [xint,wint] = JacobiGQ(alpha,beta+1,N-1)
        x = np.block( [ [-1] , [xint] ] )
        w = np.block( [ [B1] ,  [wint/(1+xint)] ] )
    if IR == 'R':
        [xint,wint] = JacobiGQ(alpha+1,beta,N-1)
        x = np.block( [ [xint] ,  [+1] ])
        w = np.block( [ [wint/(1-xint)] ,  [B2] ] )


    return x,w

# if __name__ == '__main__':
#      h = JacobiGR(-0.5,0.1,5,'R')
#      [x,w] = h
#      print(x)
#      print(w)
