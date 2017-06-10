# Modules importés

import numpy as np
import matplotlib.pyplot as plt
from tools import *
from scipy.optimize import minimize
from random import gauss
import math

# alpha : 10,10
# beta : 1
# gamma : 5, 10
# w : 5, 10

def f(alpha, beta, gamma, w):
    return np.sum((alpha-np.ones((10,10)))**2 + beta**2 + (np.dot(gamma.T, w) - np.eye(10))**2)

def df(alpha, beta, gamma, w):
    Grad = np.concatenate((2*(alpha-np.ones((10,10))).ravel(), np.array([2*beta]).ravel(), 2*w.dot(np.dot(gamma.T, w) - np.eye(10)).ravel(),\
     2*gamma.dot(np.dot(gamma.T, w) - np.eye(10)).ravel()), axis=0)
    return Grad.ravel()

alpha0 = np.zeros((10,10))
alpha = 5 * np.eye(10)
beta0 = 10
beta = 6
gamma0 = np.ones((5,10))
gamma = 2*np.ones((5,10))
w0 = np.ones((5,10)) * 0.5
w = np.ones((5,10)) * 0.9
iterr = 0
while (np.linalg.norm(alpha-alpha0)+np.linalg.norm(beta-beta0) >= 10**(-6)):
    print("ITERATION : "+str(iterr))
    iterr += 1
    alpha0 = alpha
    beta0 = beta
    gamma0 = gamma
    w0 = w
    BFGSf = lambda x : f(x[0:10*10].reshape((10,10)),float(x[10*10]), x[101:101+5*10].reshape((5,10)), x[101+5*10:101+5*10*2].reshape((5,10)))
    BFGSdf = lambda x :  df(x[0:10*10].reshape((10,10)),float(x[10*10]), x[101:101+5*10].reshape((5,10)), x[101+5*10:101+5*10*2].reshape((5,10)))

    vect_ini = np.concatenate((alpha.ravel(),np.array([beta]),gamma.ravel(),w.ravel()),axis=0)

    result = minimize(BFGSf, vect_ini, method='BFGS', jac = BFGSdf,\
                      options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})
    print(result.message)
    # print("Optimal solution : ")
    # print(result.x)
    print("Function value : ")
    print(BFGSf(result.x))
    print("Gradient value : ")
    print(BFGSdf(result.x))

    alpha = result.x[0:10*10].reshape((10,10))
    beta = float(result.x[100])
    gamma = result.x[101:101+5*10].reshape((5,10))
    w = result.x[101+5*10:101+5*10*2].reshape((5,10))

print("alpha : ")
print(alpha)
print("beta : ")
print(beta)
print("gamma : ")
print(gamma)
print("w : ")
print(w)
