from scipy.special import gamma, factorial
import numpy as np
from .JacobiGQ import JacobiGQ
from numpy import linalg as LA
def JacobiGL(alpha,beta,N):
  # Evaluate N-th order Gauss-Lobatto quadrature points x and weights w
  # associated with Jacobi polynomials of type (alpha,beta) > -1
  # note: if N is the order, length(x) = N+1
  # ES 14/10/2009 : computation of the weights using the exact formulae
  #                 (Eq.(5.15) in Hesthaven, Gottlieb, Gottlieb (2007)
  #                  and Eqs.(1.10), (2.9-10) in Gozman (2005))
  # ES 14/10/2009 : bug !!!!! with the computation of w(2) for N=2.
  # ES 16/10/2009 : corrected

  x = np.zeros((N+1,1))
  w = np.zeros((N+1,1))

  B0 = 2**(alpha+beta+1) *np.prod(np.arange(1,N))/gamma(alpha+beta+N+2)
  B1 = B0*gamma(beta +1)*gamma(beta +2)*gamma(alpha+N+1)/gamma(beta +N+1)
  B2 = B0*gamma(alpha+1)*gamma(alpha+2)*gamma(beta +N+1)/gamma(alpha+N+1)
  #
  if (N==1):
    x[0] = -1.0
    x[1] = +1.0
    w[0] =  B1
    w[1] =  B2
    return x,w
  [xint,wint] = JacobiGQ(alpha+1,beta+1,N-2)
  x = np.block( [ [-1] , [xint] , [1] ] )
  w = np.block( [ [B1] , [wint/(1-xint**2)] , [B2] ] )
  return x,w

# if __name__ == '__main__':
#      h = JacobiGL(0,0,3)
#      [x,w] = h
#      print(x)
#      print(w)
