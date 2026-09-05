import numpy as np


# Bisection method for a scalar nonlinear equation
def bisection(
    function,
    a,
    b,
    tolerance=1.0e-8,
    max_iterations=100,
):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    f_a = function(a)
    f_b = function(b)

    if f_a == 0:
        return a, 0

    if f_b == 0:
        return b, 0

    if f_a * f_b > 0:
        raise ValueError(
            "The function must change sign over the interval."
        )

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        midpoint = 0.5 * (a + b)
        f_midpoint = function(midpoint)

        if (
            abs(f_midpoint) < tolerance
            or 0.5 * abs(b - a) < tolerance
        ):
            return midpoint, iteration

        if f_a * f_midpoint < 0:
            b = midpoint
        else:
            a = midpoint
            f_a = f_midpoint

    raise RuntimeError(
        "Bisection did not converge within "
        "the maximum number of iterations."
    )


# Newton-Raphson method for a scalar nonlinear equation
def newton_raphson(
    function,
    derivative,
    initial_guess,
    tolerance=1.0e-8,
    max_iterations=100,
):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    x = float(initial_guess)

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        f_x = function(x)
        derivative_x = derivative(x)

        if abs(f_x) < tolerance:
            return x, iteration - 1

        if np.isclose(
            derivative_x,
            0.0,
        ):
            raise RuntimeError(
                "Newton-Raphson encountered a zero derivative."
            )

        x_new = x - f_x / derivative_x

        if abs(x_new - x) < tolerance:
            return x_new, iteration

        x = x_new

    raise RuntimeError(
        "Newton-Raphson did not converge within "
        "the maximum number of iterations."
    )


# Numerical Jacobian using central finite differences
def numerical_jacobian(
    function,
    x,
    step=1.0e-6,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    if step <= 0:
        raise ValueError("Step size must be positive.")

    function_value = np.asarray(
        function(x),
        dtype=float,
    )

    number_of_variables = x.size
    number_of_equations = function_value.size

    if number_of_equations != number_of_variables:
        raise ValueError(
            "The nonlinear system must contain the same "
            "number of equations and variables."
        )

    jacobian = np.zeros(
        (
            number_of_equations,
            number_of_variables,
        )
    )

    for j in range(number_of_variables):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[j] += step
        x_backward[j] -= step

        f_forward = np.asarray(
            function(x_forward),
            dtype=float,
        )

        f_backward = np.asarray(
            function(x_backward),
            dtype=float,
        )

        jacobian[:, j] = (
            f_forward - f_backward
        ) / (2.0 * step)

    return jacobian


# Newton method for a system of nonlinear equations
def newton_system(
    function,
    initial_guess,
    tolerance=1.0e-8,
    max_iterations=100,
    jacobian_step=1.0e-6,
):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    x = np.asarray(
        initial_guess,
        dtype=float,
    )

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        residual = np.asarray(
            function(x),
            dtype=float,
        )

        if np.linalg.norm(
            residual,
            ord=np.inf,
        ) < tolerance:
            return x, iteration - 1

        jacobian = numerical_jacobian(
            function,
            x,
            step=jacobian_step,
        )

        try:
            correction = np.linalg.solve(
                jacobian,
                -residual,
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "The Jacobian matrix is singular."
            ) from error

        x_new = x + correction

        if np.linalg.norm(
            x_new - x,
            ord=np.inf,
        ) < tolerance:
            return x_new, iteration

        x = x_new

    raise RuntimeError(
        "Newton's method did not converge within "
        "the maximum number of iterations."
    )