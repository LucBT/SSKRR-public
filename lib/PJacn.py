from scipy.special import gamma, factorial, comb
import numpy as np
from .JacobiP import JacobiP
from numpy import linalg as LA

def PJacn(x,NN,alpha,beta,filter_user):

# Evaluate Jacobi polynomials of type (alpha,beta) > -1 and their
# derivatives at points x up to the order NN
# Apply a filter to compute filtered Jacobi polynomials
#
# Note: they are normalized to be orthonormal
#
# Note: the computation of Q=dP/dx uses Hesthaven, Warburton:
#       Nodal Discontinuous Galerkin Methods,
#       Springer, New York (2008), Eq.(A.2) p.445.
  if np.shape(filter_user)[0] == 1:
    filter_user = np.transpose(filter_user)
  nx = int(len(x))
  N1 = NN+1
  P  = np.zeros((N1,nx))
  Q  = np.zeros((N1,nx))
  P[[0],0:nx] = filter_user[0]*np.transpose(JacobiP(x,alpha,beta,0))
  for iN in range(2,N1+1):
    norm       = np.sqrt((iN+alpha+beta)*(iN-1)) # Normalized polynomials
    # norm       = (iN+alpha+beta)./2; # Non-normalized polynomials
    P[iN-1,0:nx] = filter_user[iN-1]*JacobiP(x,alpha,beta,iN-1)
    Q[iN-1,0:nx] = norm*(filter_user[iN-1]*JacobiP(x,alpha+1,beta+1,iN-2))
  return P, Q


# if __name__ == '__main__':

#      h = PJacn([1],2,0,0,[[1],[1],[1]])
#      print(h)
