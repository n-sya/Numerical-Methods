import numpy as np

from interpolation.interpolation import gaussian_elimination

from ode.initial_value import runge_kutta_4_system

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

# Shooting method using secant iteration
def shooting_method(
    system,
    initial_state_function,
    residual_function,
    t0,
    t_end,
    h,
    first_guess,
    second_guess,
    tolerance=1.0e-8,
    max_iterations=50,
):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least 1."
        )

    def solve_with_guess(guess):
        initial_state = np.asarray(
            initial_state_function(guess),
            dtype=float,
        )

        t, solution = runge_kutta_4_system(
            system,
            initial_state,
            t0=t0,
            t_end=t_end,
            h=h,
        )

        residual = float(
            residual_function(
                t,
                solution,
            )
        )

        return (
            t,
            solution,
            residual,
        )

    guess_previous = first_guess
    guess_current = second_guess

    _, _, residual_previous = solve_with_guess(
        guess_previous
    )

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        (
            t,
            solution,
            residual_current,
        ) = solve_with_guess(
            guess_current
        )

        if abs(residual_current) < tolerance:
            return (
                t,
                solution,
                guess_current,
                iteration,
            )

        denominator = (
            residual_current
            - residual_previous
        )

        if np.isclose(
            denominator,
            0.0,
        ):
            raise RuntimeError(
                "The shooting method encountered "
                "a zero secant denominator."
            )

        guess_next = (
            guess_current
            - residual_current
            * (
                guess_current
                - guess_previous
            )
            / denominator
        )

        guess_previous = guess_current
        residual_previous = residual_current
        guess_current = guess_next

    raise RuntimeError(
        "The shooting method did not converge within "
        "the maximum number of iterations."
    )