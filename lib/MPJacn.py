from scipy.special import gamma, factorial, comb
import numpy as np
from .JacobiGQ import JacobiGQ
from .MultiIndex import MultiIndex
from .JacobiP import JacobiP
from .JNodeWt import JNodeWt
from numpy import linalg as LA



def MPJacn(x,NN,alpha,beta,Mindices):
# Evaluate Jacobi polynomial of type (alpha,beta) > -1
# at points x for order N
# Note: they are normalized to be orthonormal

# Bug !!! (ES 12/10/2009) Case alpha = beta = -0.5 for N = 0

# Turn points into row if needed
  nx = np.size(x)
  na = np.size(alpha)
  nb = np.size(beta)
  if (na!=nx) | (nb!=nx):
    print('MPJacn: the size of alpha/beta is incorrect')
    return
  N1 = NN+1
  PJ = np.zeros((N1,nx))

  for ix in range (1,nx+1):
    PJ[0,ix-1] = JacobiP(x[ix-1],alpha[ix-1],beta[ix-1],0)
    for iN in range(2,N1+1):
      norm      = np.sqrt((iN+alpha[ix-1]+beta[ix-1])*(iN-1)) # Normalized polynomials
      PJ[iN-1,ix-1] = JacobiP(x[ix-1],alpha[ix-1],beta[ix-1],iN-1)
#
# Evaluate the multivariate polynomials at x
  P1 = int(comb(NN+nx,nx))
  QJ = np.zeros((nx,1))
  XP = np.zeros((P1,1))
  Mindices = np.array(Mindices)
  for iP in range(1,P1+1):
    for ix in range(1,nx+1):
      PJ[int(Mindices[iP-1,ix-1]),ix-1]
      QJ[ix-1] = PJ[int(Mindices[iP-1,ix-1]),ix-1]
    XP[iP-1] = np.prod(QJ)

  XP = np.transpose(XP)

  return XP

#
# if __name__ == '__main__':
#      [alpha,P] = MultiIndex(8,3)
#      [pos,wei] = JNodeWt(9,2,0,0)
#      print(alpha)
#      h = MPJacn(pos,8,np.zeros((165,3)),np.zeros((165,3)),alpha)
#      print(h)
