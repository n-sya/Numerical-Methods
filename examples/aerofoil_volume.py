import csv

import matplotlib.pyplot as plt
import numpy as np

from integration.integration import trapezoidal_rule


# Define the number of points in each direction
nx = 100
ny = 15


# Create arrays for the aerofoil surface coordinates
x = np.zeros((ny, nx))
y = np.zeros((ny, nx))
z_top = np.zeros((ny, nx))
z_bottom = np.zeros((ny, nx))


# Read aerofoil surface data
with open("data/aerofoil.txt", "r") as file:
    surfaces = csv.reader(file)

    row = 0
    column = 0

    for point in surfaces:
        x[row, column] = float(point[0])
        y[row, column] = float(point[1])
        z_top[row, column] = float(point[2])
        z_bottom[row, column] = float(point[3])

        column += 1

        if column == nx:
            column = 0
            row += 1


# Integrate between the upper and lower surfaces at each spanwise location
cross_sectional_area = np.zeros(ny)

for i in range(ny):
    top_area = trapezoidal_rule(x[i, :], z_top[i, :])
    bottom_area = trapezoidal_rule(x[i, :], z_bottom[i, :])

    cross_sectional_area[i] = top_area - bottom_area


# Integrate the cross-sectional area along the span
volume = trapezoidal_rule(
    y[:, 0],
    cross_sectional_area,
)

print(f"Aerofoil volume: {volume:.6f}")


# Plot the upper and lower aerofoil surfaces
figure = plt.figure(figsize=(9, 6))
axis = figure.add_subplot(111, projection="3d")

axis.plot_surface(
    x,
    y,
    z_top,
    alpha=0.8,
)

axis.plot_surface(
    x,
    y,
    z_bottom,
    alpha=0.8,
)

axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_zlabel("z")
axis.set_title("Aerofoil Geometry")

plt.tight_layout()
plt.show()