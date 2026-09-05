import matplotlib.pyplot as plt
import numpy as np

from integration.integration import double_trapezoidal_rule


# Dome dimensions
height = 25.0
semi_major_axis = 67.0
semi_minor_axis = 56.0


# Analytical volume of the upper half of an ellipsoid
analytical_volume = (
    2.0
    / 3.0
    * np.pi
    * semi_major_axis
    * semi_minor_axis
    * height
)


# Calculate the dome height at each point
def dome_height(x_grid, y_grid):
    inside = (
        1.0
        - x_grid**2 / semi_major_axis**2
        - y_grid**2 / semi_minor_axis**2
    )

    return height * np.sqrt(
        np.maximum(
            inside,
            0.0,
        )
    )


# Perform the numerical integration for a given mesh spacing
def calculate_volume(mesh_spacing):
    x = np.arange(
        -semi_major_axis,
        semi_major_axis + mesh_spacing,
        mesh_spacing,
    )

    y = np.arange(
        -semi_minor_axis,
        semi_minor_axis + mesh_spacing,
        mesh_spacing,
    )

    x_grid, y_grid = np.meshgrid(
        x,
        y,
    )

    inside_ellipse = (
        x_grid**2 / semi_major_axis**2
        + y_grid**2 / semi_minor_axis**2
        <= 1.0
    )

    z = np.zeros_like(
        x_grid,
        dtype=float,
    )

    z[inside_ellipse] = dome_height(
        x_grid[inside_ellipse],
        y_grid[inside_ellipse],
    )

    volume = double_trapezoidal_rule(
        x,
        y,
        z,
    )

    return (
        x,
        y,
        z,
        volume,
    )


# Compare two mesh sizes
mesh_sizes = [
    0.5,
    0.1,
]

for mesh_spacing in mesh_sizes:
    _, _, _, numerical_volume = calculate_volume(
        mesh_spacing
    )

    percentage_error = (
        abs(
            numerical_volume
            - analytical_volume
        )
        / analytical_volume
        * 100.0
    )

    print(
        f"Mesh spacing: {mesh_spacing:.2f} m"
    )

    print(
        f"Numerical volume: {numerical_volume:.2f} m^3"
    )

    print(
        f"Analytical volume: {analytical_volume:.2f} m^3"
    )

    print(
        f"Percentage error: {percentage_error:.4f}%"
    )

    print()


# Plot the finer numerical surface
x, y, z, _ = calculate_volume(
    mesh_spacing=0.5
)

x_grid, y_grid = np.meshgrid(
    x,
    y,
)

fig = plt.figure(
    figsize=(9, 6)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot_surface(
    x_grid,
    y_grid,
    z,
)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("Height (m)")
ax.set_title(
    "Royal Albert Hall Dome Numerical Model"
)

plt.tight_layout()
plt.show()