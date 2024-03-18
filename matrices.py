import numpy as np
from scipy.integrate import quad

# Define the nonlinear system dynamics
def f(x):
    x1, x2 = x
    dx1dt = -x1**3 + x2
    dx2dt = x1 - x2**3
    return np.array([dx1dt, dx2dt])

# Equilibrium point
x_star = np.array([0, 0])

# Jacobian matrix of the system evaluated at x*
def J_f(x_star):
    return np.array([[0, 1], [1, 0]])

# Hessian matrix of the system evaluated at x*
def H_f(x_star):
    return np.array([[0, 0], [0, 0]])
