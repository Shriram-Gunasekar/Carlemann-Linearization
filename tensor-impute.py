import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import kronecker

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

# Tensor completion using CP decomposition (TensorLy library)
def tensor_completion(data, rank):
    factors = parafac(data, rank=rank)
    tensor_hat = tl.kruskal_to_tensor(factors)
    return tensor_hat

# Example usage with tensor completion
if __name__ == '__main__':
    # Order of Carleman linearization
    m = 2

    # Linearize the system
    A_linearized = carleman_linearization(f, J_f, H_f, x_star, m)

    # Create a tensor from the linearized matrix
    tensor_A = torch.unsqueeze(A_linearized, dim=0)

    # Fill in missing values using tensor completion
    rank = 2  # Rank of the CP decomposition
    tensor_hat = tensor_completion(tensor_A, rank)

    # Extract the completed matrix
    A_completed = tensor_hat.squeeze(0).detach().numpy()

    print("Completed system matrix:")
    print(A_completed)
