# Modules importés

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from random import gauss


def f(x):
    return (x[0]-1)**2 + (x[1]-2)**4

def df(x):
    return np.array([2*(x[0]-1),4*(x[1]-2)])


result = minimize(f, np.array([0,0]), method='BFGS', jac = df,\
options={'gtol': 1e-6, 'disp': True, 'maxiter': 2000})


print(result.x)
