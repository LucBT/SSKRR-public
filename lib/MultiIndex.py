from scipy.special import gamma, factorial, comb
import numpy as np
from .JacobiGQ import JacobiGQ
from .JacobiP import JacobiP
from numpy import linalg as LA

def MultiIndex(No,NN):

# Multi-indices of the NN-dimensional polynomials of total order No
#
# INPUT : No            = max total order of the multivariate polynomials
#         NN            = dimension
#
# OUTPUT: alpha(P+1,NN) = multi-indices
#         P             = number of multi-indices s.t.
#                         P+1 = nchoosek(No+NN,NN)
#
# After O.P. Le Maitre & O.M. Knio, "Spectral Methods for Uncertainty
# Quantification", Springer, Dordrecht (2010). pp. 516-517.
#
# v1.0 E. Savin Onera/DMNF jan. 2015
 temp = int(comb(NN+No,No))
 alpha = np.zeros((temp,NN))
 # alpha[0,:] = np.zeros((1,NN));
 if (No == 0):
   P = 0
   return alpha,P

 # alpha[1:NN+2,0:NN+1] = np.zeros((NN,NN))
 for j in range(0,NN):
   alpha[j+1,j] = 1
 if (No == 1):
   P = NN
   return alpha,P

 P = NN
 pp = np.zeros((NN,No))
 pp[0:NN,0] = 1
 for k in range(1,No):
   L = P
   for i in range(0,NN):
      pp[i,k] = pp[i:NN,k-1].sum(axis = 0)
   for j in range(0,NN):
      for m in range(L-int(pp[j,k]),L):
         P = P + 1
         alpha[P,0:NN] = alpha[m+1,0:NN]
         alpha[P,j] = alpha[P,j]+1

 # RETURN P NOT P+1
 # ALPHA RETOURNE TOUTES LES COMB POSSIBLES DES POLYNOMES DES DIFFERENTS PARAMETRES D ENTREE
 return alpha,P


# if __name__ == '__main__':
#      h = MultiIndex(2,3)
#      print(h)
