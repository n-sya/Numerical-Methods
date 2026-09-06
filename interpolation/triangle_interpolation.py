import numpy as np


# Validate the triangle vertices, nodal values, and interpolation point
def _validate_inputs(vertices, values, point):
    vertices = np.asarray(vertices, dtype=float)
    values = np.asarray(values, dtype=float)
    point = np.asarray(point, dtype=float)

    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    if values.shape != (3,):
        raise ValueError("values must contain exactly three values.")

    if point.shape != (2,):
        raise ValueError("point must contain exactly two coordinates.")

    return vertices, values, point


# Calculate barycentric coordinates for a point within a triangle
def barycentric_coordinates(vertices, point):
    vertices = np.asarray(vertices, dtype=float)
    point = np.asarray(point, dtype=float)

    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")

    if point.shape != (2,):
        raise ValueError("point must contain exactly two coordinates.")

    coordinate_matrix = np.vstack((vertices.T, np.ones(3)))
    point_vector = np.append(point, 1.0)

    try:
        return np.linalg.solve(coordinate_matrix, point_vector)
    except np.linalg.LinAlgError as error:
        raise ValueError("The triangle vertices must not be collinear.") from error


# Interpolate a value using barycentric coordinates
def barycentric_interpolation(vertices, values, point):
    vertices, values, point = _validate_inputs(vertices, values, point)

    coordinates = barycentric_coordinates(vertices, point)

    return np.dot(coordinates, values)


# Interpolate a value using inverse-distance weighting
def inverse_distance_interpolation(vertices, values, point):
    vertices, values, point = _validate_inputs(vertices, values, point)

    distances = np.linalg.norm(vertices - point, axis=1)

    matching_vertices = np.isclose(distances, 0.0)

    if np.any(matching_vertices):
        vertex_index = np.flatnonzero(matching_vertices)[0]
        return values[vertex_index]

    weights = 1.0 / distances

    return np.dot(weights, values) / np.sum(weights)