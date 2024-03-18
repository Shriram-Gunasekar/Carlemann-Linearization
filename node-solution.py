import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchdiffeq import odeint

# Define the nonlinear system dynamics function f(x)
def f(x):
    x1, x2 = x
    dx1dt = -x1**3 + x2
    dx2dt = x1 - x2**3
    return torch.tensor([dx1dt, dx2dt])

# Define the equilibrium point x*
x_star = torch.tensor([0., 0.])

# Define the Jacobian matrix of the system evaluated at x*
def J_f(x_star):
    return torch.tensor([[0., 1.], [1., 0.]])

# Define the Hessian matrix of the system evaluated at x*
def H_f(x_star):
    return torch.tensor([[0., 0.], [0., 0.]])

# Carleman linearization term for a vector-valued function g(x)
def C_p(g, x_star, p):
    def integrand(t):
        return (1 - t)**(p-1) * torch.matmul(torch.matrix_power(g(x_star), p), g(x_star * t + (1 - t) * x_star))

    result, _ = torch.quad(integrand, torch.tensor(0.), torch.tensor(1.))
    return result / p

# Perform Carleman linearization up to order m
def carleman_linearization(f, J_f, H_f, x_star, m):
    A_linearized = torch.zeros((len(x_star), len(x_star)))

    for p in range(1, m + 1):
        A_linearized += C_p(lambda x: torch.matmul(torch.matrix_power(J_f(x), p), f(x)), x_star, p)

        if p > 1:
            for q in range(2, p + 1):
                A_linearized -= C_p(lambda x: torch.matmul(torch.matrix_power(J_f(x), p - q + 1), torch.matmul(H_f(x), torch.matrix_power(J_f(x), q - 1))), x_star, p)

    return A_linearized

# Define the Neural ODE model using PyTorch
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, t, x):
        return self.func(x)

# Example usage with Neural ODE
if __name__ == '__main__':
    # Order of Carleman linearization
    m = 2

    # Linearize the system
    A_linearized = carleman_linearization(f, J_f, H_f, x_star, m)

    # Define the Neural ODE model
    model = NeuralODE(lambda x: torch.matmul(A_linearized, x))

    # Initial condition and time points for integration
    x0 = torch.tensor([1., -1.])
    t = torch.linspace(0., 10., 1000)

    # Integrate the Neural ODE
    solution = odeint(model, x0, t)

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(t, solution[:, 0].detach().numpy(), label='x1 (Neural ODE)')
    plt.plot(t, solution[:, 1].detach().numpy(), label='x2 (Neural ODE)')
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.legend()
    plt.grid(True)
    plt.title('Neural ODE Solution')
    plt.show()
