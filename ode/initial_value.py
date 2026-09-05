import numpy as np


# Create a uniformly spaced time grid
def _create_time_grid(t0, t_end, h):
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    if t_end <= t0:
        raise ValueError("t_end must be greater than t0.")

    number_of_steps = (t_end - t0) / h

    if not np.isclose(number_of_steps, round(number_of_steps)):
        raise ValueError(
            "The time interval must contain an integer number of steps."
        )

    number_of_steps = int(round(number_of_steps))

    return np.linspace(
        t0,
        t_end,
        number_of_steps + 1,
    )


# Forward Euler method for a first-order ODE
def forward_euler(function, t0, y0, h, t_end):
    t = _create_time_grid(t0, t_end, h)

    y = np.zeros(t.size, dtype=float)
    y[0] = y0

    for n in range(1, t.size):
        y[n] = y[n - 1] + h * function(
            t[n - 1],
            y[n - 1],
        )

    return t, y


# Forward Euler method for a system of first-order ODEs
def forward_euler_system(function, y0, t0, t_end, h):
    t = _create_time_grid(t0, t_end, h)

    y0 = np.asarray(y0, dtype=float)

    number_of_variables = y0.size
    number_of_steps = t.size

    solution = np.zeros(
        (
            number_of_variables,
            number_of_steps,
        )
    )

    solution[:, 0] = y0

    for n in range(1, number_of_steps):
        derivative = np.asarray(
            function(
                t[n - 1],
                solution[:, n - 1],
            ),
            dtype=float,
        )

        if derivative.size != number_of_variables:
            raise ValueError(
                "The derivative function must return one value "
                "for each dependent variable."
            )

        solution[:, n] = (
            solution[:, n - 1]
            + h * derivative
        )

    return t, solution


# Fourth-order Runge-Kutta method for a first-order ODE
def runge_kutta_4(function, t0, y0, h, t_end):
    t = _create_time_grid(t0, t_end, h)

    y = np.zeros(t.size, dtype=float)
    y[0] = y0

    for n in range(1, t.size):
        current_t = t[n - 1]
        current_y = y[n - 1]

        k1 = function(
            current_t,
            current_y,
        )

        k2 = function(
            current_t + 0.5 * h,
            current_y + 0.5 * h * k1,
        )

        k3 = function(
            current_t + 0.5 * h,
            current_y + 0.5 * h * k2,
        )

        k4 = function(
            current_t + h,
            current_y + h * k3,
        )

        y[n] = current_y + (h / 6) * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )

    return t, y


# Fourth-order Runge-Kutta method for a system of ODEs
def runge_kutta_4_system(function, y0, t0, t_end, h):
    t = _create_time_grid(t0, t_end, h)

    y0 = np.asarray(y0, dtype=float)

    number_of_variables = y0.size
    number_of_steps = t.size

    solution = np.zeros(
        (
            number_of_variables,
            number_of_steps,
        )
    )

    solution[:, 0] = y0

    for n in range(1, number_of_steps):
        current_t = t[n - 1]
        current_y = solution[:, n - 1]

        k1 = np.asarray(
            function(
                current_t,
                current_y,
            ),
            dtype=float,
        )

        k2 = np.asarray(
            function(
                current_t + 0.5 * h,
                current_y + 0.5 * h * k1,
            ),
            dtype=float,
        )

        k3 = np.asarray(
            function(
                current_t + 0.5 * h,
                current_y + 0.5 * h * k2,
            ),
            dtype=float,
        )

        k4 = np.asarray(
            function(
                current_t + h,
                current_y + h * k3,
            ),
            dtype=float,
        )

        solution[:, n] = current_y + (h / 6) * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )

    return t, solution


# Backward Euler method for a scalar ODE using Newton iteration
def backward_euler(
    function,
    derivative_y,
    t0,
    y0,
    h,
    t_end,
    tolerance=1.0e-10,
    max_iterations=50,
):
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be at least 1."
        )

    t = _create_time_grid(t0, t_end, h)

    y = np.zeros(t.size, dtype=float)
    y[0] = y0

    for n in range(1, t.size):
        previous_y = y[n - 1]

        # Use Forward Euler as the initial guess
        current_y = previous_y + h * function(
            t[n - 1],
            previous_y,
        )

        for _ in range(max_iterations):
            residual = (
                current_y
                - previous_y
                - h * function(
                    t[n],
                    current_y,
                )
            )

            residual_derivative = (
                1
                - h
                * derivative_y(
                    t[n],
                    current_y,
                )
            )

            if np.isclose(
                residual_derivative,
                0.0,
            ):
                raise RuntimeError(
                    "Newton iteration encountered "
                    "a zero derivative."
                )

            new_y = (
                current_y
                - residual / residual_derivative
            )

            if abs(new_y - current_y) < tolerance:
                current_y = new_y
                break

            current_y = new_y

        else:
            raise RuntimeError(
                "Backward Euler did not converge within "
                "the maximum number of iterations."
            )

        y[n] = current_y

    return t, y