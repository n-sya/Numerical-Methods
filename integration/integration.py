import numpy as np


# Composite trapezoidal rule for evenly or unevenly spaced nodes
def trapezoidal_rule(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must contain the same number of values.")

    if x.size < 2:
        raise ValueError("At least two integration nodes are required.")

    integral = 0.0

    for i in range(x.size - 1):
        integral += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])

    return integral


# Composite Simpson's 1/3 rule for evenly spaced nodes
def simpson_rule(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must contain the same number of values.")

    if x.size < 3:
        raise ValueError("At least three integration nodes are required.")

    number_of_subintervals = x.size - 1

    if number_of_subintervals % 2 != 0:
        raise ValueError(
            "Simpson's rule requires an even number of subintervals."
        )

    spacing = np.diff(x)

    if not np.allclose(spacing, spacing[0]):
        raise ValueError("Simpson's rule requires equally spaced nodes.")

    h = spacing[0]

    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])
    integral += 2 * np.sum(y[2:-1:2])

    return h * integral / 3


# Adaptive Simpson integration using successive grid refinement
def adaptive_simpson(
    func,
    a,
    b,
    tolerance=1.0e-8,
    max_iterations=20,
):
    if b <= a:
        raise ValueError(
            "The upper integration limit must be greater than the lower limit."
        )

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    number_of_subintervals = 2

    x = np.linspace(a, b, number_of_subintervals + 1)
    y = func(x)

    previous_integral = simpson_rule(x, y)

    for _ in range(max_iterations):
        number_of_subintervals *= 2

        x = np.linspace(a, b, number_of_subintervals + 1)
        y = func(x)

        current_integral = simpson_rule(x, y)

        estimated_error = abs(
            current_integral - previous_integral
        ) / 15

        if estimated_error < tolerance:
            return (
                current_integral,
                estimated_error,
                number_of_subintervals,
            )

        previous_integral = current_integral

    raise RuntimeError(
        "Adaptive Simpson integration did not converge within "
        "the maximum number of iterations."
    )


# Gauss-Legendre quadrature using between one and five nodes
def gauss_legendre_quadrature(
    func,
    a,
    b,
    number_of_nodes,
):
    if b <= a:
        raise ValueError(
            "The upper integration limit must be greater than the lower limit."
        )

    if number_of_nodes < 1 or number_of_nodes > 5:
        raise ValueError(
            "number_of_nodes must be between 1 and 5."
        )

    nodes = [
        np.array(
            [
                0.0,
            ]
        ),
        np.array(
            [
                -1 / np.sqrt(3),
                1 / np.sqrt(3),
            ]
        ),
        np.array(
            [
                -np.sqrt(3 / 5),
                0.0,
                np.sqrt(3 / 5),
            ]
        ),
        np.array(
            [
                -np.sqrt(
                    3 / 7
                    + (2 / 7) * np.sqrt(6 / 5)
                ),
                -np.sqrt(
                    3 / 7
                    - (2 / 7) * np.sqrt(6 / 5)
                ),
                np.sqrt(
                    3 / 7
                    - (2 / 7) * np.sqrt(6 / 5)
                ),
                np.sqrt(
                    3 / 7
                    + (2 / 7) * np.sqrt(6 / 5)
                ),
            ]
        ),
        np.array(
            [
                -(1 / 3)
                * np.sqrt(
                    5
                    + 2 * np.sqrt(10 / 7)
                ),
                -(1 / 3)
                * np.sqrt(
                    5
                    - 2 * np.sqrt(10 / 7)
                ),
                0.0,
                (1 / 3)
                * np.sqrt(
                    5
                    - 2 * np.sqrt(10 / 7)
                ),
                (1 / 3)
                * np.sqrt(
                    5
                    + 2 * np.sqrt(10 / 7)
                ),
            ]
        ),
    ]

    weights = [
        np.array(
            [
                2.0,
            ]
        ),
        np.array(
            [
                1.0,
                1.0,
            ]
        ),
        np.array(
            [
                5 / 9,
                8 / 9,
                5 / 9,
            ]
        ),
        np.array(
            [
                (18 - np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 - np.sqrt(30)) / 36,
            ]
        ),
        np.array(
            [
                (322 - 13 * np.sqrt(70)) / 900,
                (322 + 13 * np.sqrt(70)) / 900,
                128 / 225,
                (322 + 13 * np.sqrt(70)) / 900,
                (322 - 13 * np.sqrt(70)) / 900,
            ]
        ),
    ]

    selected_nodes = nodes[number_of_nodes - 1]
    selected_weights = weights[number_of_nodes - 1]

    midpoint = 0.5 * (a + b)
    half_width = 0.5 * (b - a)

    integral = 0.0

    for node, weight in zip(
        selected_nodes,
        selected_weights,
    ):
        transformed_x = midpoint + half_width * node
        integral += weight * func(transformed_x)

    return half_width * integral