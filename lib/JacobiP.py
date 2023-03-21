from scipy.special import gamma, factorial
import numpy as np
from numpy import linalg as LA
def JacobiP(x,alpha,beta,N):
  # Evaluate Jacobi polynomial of type (alpha,beta) > -1
  # at points x for order N
  # Note: they are normalized to be orthonormal

  # Bug !!! (ES 12/10/2009) Case alpha = beta = -0.5 for N = 0

  # Turn points into row if needed
  xp = x
  # xp = np.reshape(xp,(1,2))
  # dims = np.shape(xp)
  # print(dims)
  # if (dims[1] == 1):
  #   xp = np.transpose(xp)
  try:
    NUM_OBS = np.shape(xp)[0] # Multiple points
  except:
    NUM_OBS= 1 # Only one point

  PL = np.zeros((N+1, NUM_OBS)) # N+1 polynome car commence au polynome de degré 0
  # PL = np.zeros((N+1, 1)) # N+1 polynome car commence au polynome de degré 0
  xp = np.array(xp)
  # Initial values P_0(x) ans P_1(x)
  if (alpha ==-0.5 and beta ==-0.5):
    gamma0 = np.pi
  else:
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)* \
          gamma(beta+1)/gamma(alpha+beta+1)

  PL[0,:] = 1.0/np.sqrt(gamma0)
  if (N==0):
    P = np.transpose(PL[N,:])
    return PL
  #
  gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
  PL[1,:] = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/np.sqrt(gamma1)
  if (N==1):
    P = np.transpose(PL[N,:])
    return P
  # Repeat value in recurrence
  aold = 2/(alpha+beta+2)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
  # Forward recurrence using the symmetry of the recurrence
  for i in range(1, N):
    h1 = 2*i + alpha + beta
    anew = 2/(h1+2)*np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)* \
            (i+1+beta)/(h1+1)/(h1+3))
    bnew = -(alpha**2-beta**2)/h1/(h1+2) # cn in Computation of higher-order moments of generalized polynomial chaos expansions (Savin,Faverjon, 2017 )
    PL[i+1,:] = 1/anew*(-aold*PL[i-1,:] + (xp-bnew)*PL[i,:])
    aold = anew
  # PL DONNE LES VALEURS DES POLYNOMES (DEGRE 0 À N) OU LES LIGNES DONNENT LA VALEUR DU POLY DE DEGRE ORDRE CROISSANT ET LES COLONNES AUX DIFFERENTS POINTS X
  # print(PL)
  P = np.transpose(PL[N,:]) # valeur du polynome au point x de degré N
  return P


# if __name__ == '__main__':
#      h = JacobiP([2.4],0,0,0)
#      print(h)
