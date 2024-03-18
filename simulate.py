import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define the nonlinear system dynamics function f(x)
def f(x):
    # Modify this function according to your specific nonlinear system
    x1, x2 = x
    dx1dt = -x1**3 + x2
    dx2dt = x1 - x2**3
    return np.array([dx1dt, dx2dt])

# Define the equilibrium point x*
x_star = np.array([0, 0])

# Define the Jacobian matrix of the system evaluated at x*
def J_f(x_star):
    # Modify this function to compute the Jacobian matrix of f(x) at x*
    return np.array([[0, 1], [1, 0]])

# Define the Hessian matrix of the system evaluated at x*
def H_f(x_star):
    # Modify this function to compute the Hessian matrix of f(x) at x*
    return np.array([[0, 0], [0, 0]])

# Carleman linearization term for a vector-valued function g(x)
def C_p(g, x_star, p):
    def integrand(t):
        return (1 - t)**(p-1) * np.dot(np.linalg.matrix_power(g(x_star), p), g(x_star * t + (1 - t) * x_star))

    result, _ = quad(integrand, 0, 1)
    return result / p

# Perform Carleman linearization up to order m
def carleman_linearization(f, J_f, H_f, x_star, m):
    A_linearized = np.zeros((len(x_star), len(x_star)))

    for p in range(1, m + 1):
        A_linearized += C_p(lambda x: np.dot(np.linalg.matrix_power(J_f(x), p), f(x)), x_star, p)

        if p > 1:
            for q in range(2, p + 1):
                A_linearized -= C_p(lambda x: np.dot(np.linalg.matrix_power(J_f(x), p - q + 1), np.dot(H_f(x), np.linalg.matrix_power(J_f(x), q - 1))), x_star, p)

    return A_linearized

# Example usage
m = 2  # Order of Carleman linearization
A_linearized = carleman_linearization(f, J_f, H_f, x_star, m)

print("Linearized system matrix:")
print(A_linearized)

# Visualization: Plotting the nonlinear system and its linearized approximation
t = np.linspace(0, 10, 1000)
x0 = np.array([1, -1])  # Initial condition

# Simulate the nonlinear system
def simulate_nonlinear_system(f, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x[i, :] = x[i - 1, :] + f(x[i - 1, :]) * dt
    return x

# Simulate the linearized system
def simulate_linearized_system(A, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x[i, :] = x[i - 1, :] + np.dot(A, x[i - 1, :]) * dt
    return x

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# Nonlinear system simulation
x_nonlinear = simulate_nonlinear_system(f, x0, t)
ax[0].plot(t, x_nonlinear[:, 0], label='x1 (nonlinear)')
ax[0].plot(t, x_nonlinear[:, 1], label='x2 (nonlinear)')

# Linearized system simulation
x_linearized = simulate_linearized_system(A_linearized, x0, t)
ax[1].plot(t, x_linearized[:, 0], label='x1 (linearized)')
ax[1].plot(t, x_linearized[:, 1], label='x2 (linearized)')

# Plot settings
for axis in ax:
    axis.legend()
    axis.grid(True)
    axis.set_xlabel('Time')
ax[0].set_ylabel('State Variables (Nonlinear)')
ax[1].set_ylabel('State Variables (Linearized)')

plt.tight_layout()
plt.show()
