import numpy as np


# Gaussian elimination with back substitution
def gaussian_elimination(A, b):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    if b.size != A.shape[0]:
        raise ValueError("b must contain one value for each row of A.")

    number_of_equations = len(b)

    for i in range(number_of_equations - 1):
        if np.isclose(A[i, i], 0.0):
            raise ValueError(
                "Zero pivot encountered during Gaussian elimination."
            )

        for j in range(i + 1, number_of_equations):
            factor = A[j, i] / A[i, i]
            A[j, :] -= factor * A[i, :]
            b[j] -= factor * b[i]

    if np.isclose(A[-1, -1], 0.0):
        raise ValueError(
            "Zero pivot encountered during Gaussian elimination."
        )

    x = np.zeros(number_of_equations)

    for i in range(number_of_equations - 1, -1, -1):
        x[i] = b[i]

        for j in range(i + 1, number_of_equations):
            x[i] -= A[i, j] * x[j]

        x[i] /= A[i, i]

    return x


# Lagrange basis polynomial
def lagrange_basis(index, x_nodes, x):
    x_nodes = np.asarray(x_nodes, dtype=float)

    basis = 1.0

    for k in range(x_nodes.size):
        if k != index:
            denominator = x_nodes[index] - x_nodes[k]

            if np.isclose(denominator, 0.0):
                raise ValueError("Interpolation nodes must be unique.")

            basis *= (x - x_nodes[k]) / denominator

    return basis


# Lagrange polynomial interpolation
def lagrange_interpolation(x_nodes, y_nodes, x_points):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_points = np.asarray(x_points, dtype=float)

    if x_nodes.size != y_nodes.size:
        raise ValueError(
            "x_nodes and y_nodes must have the same length."
        )

    if x_nodes.size < 2:
        raise ValueError(
            "At least two interpolation nodes are required."
        )

    y_points = np.zeros_like(x_points, dtype=float)

    for i, x in enumerate(x_points):
        interpolated_value = 0.0

        for j in range(x_nodes.size):
            interpolated_value += y_nodes[j] * lagrange_basis(
                j, x_nodes, x
            )

        y_points[i] = interpolated_value

    return y_points


# Newton divided difference using recursion
def newton_divided_difference_recursive(x_nodes, y_nodes):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.size != y_nodes.size:
        raise ValueError(
            "x_nodes and y_nodes must have the same length."
        )

    if x_nodes.size == 1:
        return y_nodes[0]

    denominator = x_nodes[-1] - x_nodes[0]

    if np.isclose(denominator, 0.0):
        raise ValueError("Interpolation nodes must be unique.")

    upper_difference = newton_divided_difference_recursive(
        x_nodes[1:], y_nodes[1:]
    )

    lower_difference = newton_divided_difference_recursive(
        x_nodes[:-1], y_nodes[:-1]
    )

    return (upper_difference - lower_difference) / denominator


# Newton divided difference using an iterative method
def newton_divided_difference_iterative(x_nodes, y_nodes):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.size != y_nodes.size:
        raise ValueError(
            "x_nodes and y_nodes must have the same length."
        )

    differences = np.copy(y_nodes)
    number_of_nodes = x_nodes.size

    for order in range(number_of_nodes - 1):
        for i in range(number_of_nodes - order - 1):
            denominator = x_nodes[i + order + 1] - x_nodes[i]

            if np.isclose(denominator, 0.0):
                raise ValueError("Interpolation nodes must be unique.")

            differences[i] = (
                differences[i + 1] - differences[i]
            ) / denominator

    return differences[0]


# Newton polynomial interpolation
def newton_interpolation(x_nodes, y_nodes, x_points):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_points = np.asarray(x_points, dtype=float)

    if x_nodes.size != y_nodes.size:
        raise ValueError(
            "x_nodes and y_nodes must have the same length."
        )

    if x_nodes.size < 2:
        raise ValueError(
            "At least two interpolation nodes are required."
        )

    y_points = np.zeros_like(x_points, dtype=float)

    for point_index, x in enumerate(x_points):
        interpolated_value = y_nodes[0]

        for order in range(1, x_nodes.size):
            product = 1.0

            for j in range(order):
                product *= x - x_nodes[j]

            coefficient = newton_divided_difference_iterative(
                x_nodes[: order + 1],
                y_nodes[: order + 1],
            )

            interpolated_value += coefficient * product

        y_points[point_index] = interpolated_value

    return y_points


# Cubic spline interpolation with specified endpoint gradients
def cubic_spline(
    x_nodes,
    y_nodes,
    lower_gradient,
    upper_gradient,
    x_points,
):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    x_points = np.asarray(x_points, dtype=float)

    if x_nodes.size != y_nodes.size:
        raise ValueError(
            "x_nodes and y_nodes must have the same length."
        )

    if x_nodes.size < 2:
        raise ValueError(
            "At least two interpolation nodes are required."
        )

    if np.any(np.diff(x_nodes) <= 0):
        raise ValueError("x_nodes must be strictly increasing.")

    number_of_nodes = x_nodes.size
    number_of_splines = number_of_nodes - 1

    a = np.zeros(number_of_splines)
    b = np.zeros(number_of_splines)
    c = np.zeros(number_of_splines)
    d = np.zeros(number_of_splines)

    matrix = np.zeros((number_of_nodes, number_of_nodes))
    vector = np.zeros(number_of_nodes)

    matrix[0, 0] = 1.0
    vector[0] = lower_gradient

    matrix[-1, -1] = 1.0
    vector[-1] = upper_gradient

    for j in range(1, number_of_splines):
        h_left = x_nodes[j] - x_nodes[j - 1]
        h_right = x_nodes[j + 1] - x_nodes[j]

        matrix[j, j - 1] = 1 / h_left
        matrix[j, j] = 2 / h_left + 2 / h_right
        matrix[j, j + 1] = 1 / h_right

        vector[j] = 3 * (
            (y_nodes[j] - y_nodes[j - 1]) / h_left**2
            + (y_nodes[j + 1] - y_nodes[j]) / h_right**2
        )

    gradients = gaussian_elimination(matrix, vector)

    for j in range(number_of_splines):
        h = x_nodes[j + 1] - x_nodes[j]

        a[j] = y_nodes[j]
        b[j] = gradients[j]

        c[j] = (
            3 * (y_nodes[j + 1] - y_nodes[j]) / h**2
            - (gradients[j + 1] + 2 * gradients[j]) / h
        )

        d[j] = (
            -2 * (y_nodes[j + 1] - y_nodes[j]) / h**3
            + (gradients[j + 1] + gradients[j]) / h**2
        )

    y_points = np.full_like(x_points, np.nan, dtype=float)

    for j in range(number_of_splines):
        mask = (x_points >= x_nodes[j]) & (
            x_points <= x_nodes[j + 1]
        )

        local_x = x_points[mask] - x_nodes[j]

        y_points[mask] = (
            a[j]
            + b[j] * local_x
            + c[j] * local_x**2
            + d[j] * local_x**3
        )

    return y_points