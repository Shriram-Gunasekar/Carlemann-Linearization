# Perform Carleman linearization up to order m
def carleman_linearization(f, J_f, H_f, x_star, m):
    A_linearized = np.zeros((2, 2))

    # Linear term (order 1)
    A_linearized += J_f(x_star)

    # Quadratic term (order 2)
    A_linearized += C_p(lambda x: np.dot(H_f(x_star), f(x)), x_star, 2)

    return A_linearized

# Compute the linearized system matrix
A_linearized = carleman_linearization(f, J_f, H_f, x_star, m=2)

print("Linearized system matrix:")
print(A_linearized)
