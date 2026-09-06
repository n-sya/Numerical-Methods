import matplotlib.pyplot as plt
import numpy as np

from interpolation.triangle_interpolation import (
    barycentric_interpolation,
    inverse_distance_interpolation,
)


# Define a triangular element
vertices = np.array([[-1.0, -1.0], [1.0, 3.0], [2.0, 0.5]])
values = np.array([1.0, 1.5, 3.0])


# Generate interpolation points inside the triangle
number_of_points = 80

x_values = np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), number_of_points)
y_values = np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), number_of_points)

x_grid, y_grid = np.meshgrid(x_values, y_values)

points = np.column_stack((x_grid.ravel(), y_grid.ravel()))


# Keep only points inside the triangle using barycentric coordinates
coordinate_matrix = np.vstack((vertices.T, np.ones(3)))
point_matrix = np.vstack((points.T, np.ones(points.shape[0])))

barycentric_coordinates = np.linalg.solve(coordinate_matrix, point_matrix)

inside_triangle = np.all(barycentric_coordinates >= -1e-12, axis=0)

interpolation_points = points[inside_triangle]


# Calculate both interpolation methods
barycentric_values = np.array(
    [
        barycentric_interpolation(vertices, values, point)
        for point in interpolation_points
    ]
)

inverse_distance_values = np.array(
    [
        inverse_distance_interpolation(vertices, values, point)
        for point in interpolation_points
    ]
)


# Calculate the difference between the methods
difference = np.abs(barycentric_values - inverse_distance_values)

print(f"Maximum interpolation difference: {np.max(difference):.4f}")
print(f"Mean interpolation difference: {np.mean(difference):.4f}")


# Plot barycentric interpolation
plt.figure(figsize=(8, 6))

contour = plt.tricontourf(
    interpolation_points[:, 0],
    interpolation_points[:, 1],
    barycentric_values,
    levels=20,
)

plt.colorbar(contour, label="Interpolated value")

plt.plot(
    np.append(vertices[:, 0], vertices[0, 0]),
    np.append(vertices[:, 1], vertices[0, 1]),
    "k-",
)

plt.scatter(vertices[:, 0], vertices[:, 1])

plt.xlabel("x")
plt.ylabel("y")
plt.title("Barycentric Triangle Interpolation")

plt.tight_layout()
plt.savefig(
    "outputs/barycentric_triangle_interpolation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot inverse-distance interpolation
plt.figure(figsize=(8, 6))

contour = plt.tricontourf(
    interpolation_points[:, 0],
    interpolation_points[:, 1],
    inverse_distance_values,
    levels=20,
)

plt.colorbar(contour, label="Interpolated value")

plt.plot(
    np.append(vertices[:, 0], vertices[0, 0]),
    np.append(vertices[:, 1], vertices[0, 1]),
    "k-",
)

plt.scatter(vertices[:, 0], vertices[:, 1])

plt.xlabel("x")
plt.ylabel("y")
plt.title("Inverse-Distance Triangle Interpolation")

plt.tight_layout()
plt.savefig(
    "outputs/inverse_distance_triangle_interpolation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot the difference between interpolation methods
plt.figure(figsize=(8, 6))

contour = plt.tricontourf(
    interpolation_points[:, 0],
    interpolation_points[:, 1],
    difference,
    levels=20,
)

plt.colorbar(contour, label="Absolute difference")

plt.plot(
    np.append(vertices[:, 0], vertices[0, 0]),
    np.append(vertices[:, 1], vertices[0, 1]),
    "k-",
)

plt.scatter(vertices[:, 0], vertices[:, 1])

plt.xlabel("x")
plt.ylabel("y")
plt.title("Difference Between Triangle Interpolation Methods")

plt.tight_layout()
plt.savefig(
    "outputs/triangle_interpolation_difference.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()