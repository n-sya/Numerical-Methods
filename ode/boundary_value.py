import numpy as np

from interpolation.interpolation import gaussian_elimination


# Finite-difference solution of a second-order linear ODE
# y'' + f(x)y' + g(x)y = p(x)
# with Dirichlet boundary conditions
def finite_difference_dirichlet(
    coefficient_function,
    a,
    b,
    y_a,
    y_b,
    number_of_intervals,
):
    if b <= a:
        raise ValueError(
            "The upper boundary must be greater than the lower boundary."
        )

    if number_of_intervals < 2:
        raise ValueError(
            "At least two intervals are required."
        )

    x = np.linspace(
        a,
        b,
        number_of_intervals + 1,
    )

    h = x[1] - x[0]

    matrix = np.zeros(
        (
            number_of_intervals + 1,
            number_of_intervals + 1,
        )
    )

    vector = np.zeros(
        number_of_intervals + 1
    )

    # Left boundary condition
    matrix[0, 0] = 1.0
    vector[0] = y_a

    # Right boundary condition
    matrix[-1, -1] = 1.0
    vector[-1] = y_b

    # Interior finite-difference equations
    for i in range(
        1,
        number_of_intervals,
    ):
        f, g, p = coefficient_function(x[i])

        matrix[i, i - 1] = (
            1 / h**2
            - f / (2 * h)
        )

        matrix[i, i] = (
            g
            - 2 / h**2
        )

        matrix[i, i + 1] = (
            1 / h**2
            + f / (2 * h)
        )

        vector[i] = p

    y = gaussian_elimination(
        matrix,
        vector,
    )

    return x, y


# Finite-difference solution of a second-order linear ODE
# with general Robin boundary conditions
#
# left:
# alpha_a*y'(a) + beta_a*y(a) = gamma_a
#
# right:
# alpha_b*y'(b) + beta_b*y(b) = gamma_b
def finite_difference_robin(
    coefficient_function,
    a,
    b,
    left_boundary,
    right_boundary,
    number_of_intervals,
):
    if b <= a:
        raise ValueError(
            "The upper boundary must be greater than the lower boundary."
        )

    if number_of_intervals < 2:
        raise ValueError(
            "At least two intervals are required."
        )

    alpha_a, beta_a, gamma_a = left_boundary
    alpha_b, beta_b, gamma_b = right_boundary

    x = np.linspace(
        a,
        b,
        number_of_intervals + 1,
    )

    h = x[1] - x[0]

    matrix = np.zeros(
        (
            number_of_intervals + 1,
            number_of_intervals + 1,
        )
    )

    vector = np.zeros(
        number_of_intervals + 1
    )

    # Forward difference at the left boundary
    matrix[0, 0] = (
        beta_a
        - alpha_a / h
    )

    matrix[0, 1] = alpha_a / h

    vector[0] = gamma_a

    # Backward difference at the right boundary
    matrix[-1, -2] = -alpha_b / h

    matrix[-1, -1] = (
        alpha_b / h
        + beta_b
    )

    vector[-1] = gamma_b

    # Interior finite-difference equations
    for i in range(
        1,
        number_of_intervals,
    ):
        f, g, p = coefficient_function(x[i])

        matrix[i, i - 1] = (
            1 / h**2
            - f / (2 * h)
        )

        matrix[i, i] = (
            g
            - 2 / h**2
        )

        matrix[i, i + 1] = (
            1 / h**2
            + f / (2 * h)
        )

        vector[i] = p

    y = gaussian_elimination(
        matrix,
        vector,
    )

    return x, y