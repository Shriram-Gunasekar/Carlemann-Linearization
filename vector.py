# Carleman linearization term for a vector-valued function g(x)
def C_p(g, x_star, p):
    def integrand(t):
        return (1 - t)**(p-1) * np.dot(np.linalg.matrix_power(g(x_star), p), g(x_star * t + (1 - t) * x_star))

    result, _ = quad(integrand, 0, 1)
    return result / p
