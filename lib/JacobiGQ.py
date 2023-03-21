from scipy.special import gamma, factorial
import numpy as np
from numpy import linalg as LA
def JacobiGQ(alpha,beta,N):
# Evaluate N-th order Gauss quadrature points x and weights w
# associated with Jacobi polynomials of type (alpha,beta) > -1
# Note: if N is the order, length(x) = N+1
# After Hesthaven, Warburton (2008), pp.447-448
  if (N==0):
    x = np.zeros((1,1))
    w = np.zeros((1,1))
    x[0,0] = (beta-alpha)/(alpha+beta+2)
    w[0,0] = 2**(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+2)
  #Form symmetric matrix for recurrence
  else:
    J  = np.zeros(N+1)
    h1 = 2*np.arange(0,N+1) + alpha + beta
    J1 = np.zeros((1,N+1))
    J1[0,0]= (-1/2*(alpha-beta)/(h1[0]+2))
    J1[0,1:] = -1/2*(alpha**2-beta**2)/(h1[1:(N+1)]+2)/h1[1:(N+1)]
    J  = np.diag(J1[0]) + \
      np.diag(2/(h1[0:N]+2)*np.sqrt(np.arange(1,N+1)*(np.arange(1,N+1)+alpha+beta)* \
      (np.arange(1,N+1)+alpha)*(np.arange(1,N+1)+beta)/(h1[0:N]+1)/(h1[0:N]+3)),1)

    J = J+np.transpose(J)
  # Compute quadrature by eigenvalue solve
    [D,V] = LA.eigh(J)
    x = np.transpose([D])
    w = np.transpose([(np.transpose(V[0,:]))**2*2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
            gamma(beta+1)/gamma(alpha+beta+1)])
  return x,w

# if __name__ == '__main__':
#      h = JacobiGQ(-0.5,0.1,0)
#      [x,w] = h
#      print(x)
#      print(w)
