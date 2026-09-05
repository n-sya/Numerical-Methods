import matplotlib.pyplot as plt
import numpy as np

from interpolation.interpolation import lagrange_interpolation


# Function to interpolate
def function(x):
    return np.sin(x)


# Evaluation points
x_points = np.linspace(0.0, 3.0, 500)
y_exact = function(x_points)

# Node counts corresponding to different polynomial degrees
node_counts = [2, 3, 4]


# Plot the exact function
plt.figure(figsize=(8, 6))

plt.plot(
    x_points,
    y_exact,
    linewidth=2,
    label="Exact Function",
)


# Construct interpolating polynomials
for number_of_nodes in node_counts:
    x_nodes = np.linspace(
        1.0,
        2.0,
        number_of_nodes,
    )
    y_nodes = function(x_nodes)

    y_interpolated = lagrange_interpolation(
        x_nodes,
        y_nodes,
        x_points,
    )

    polynomial_degree = number_of_nodes - 1

    plt.plot(
        x_points,
        y_interpolated,
        label=f"Degree {polynomial_degree}",
    )

    plt.scatter(
        x_nodes,
        y_nodes,
        s=30,
    )


plt.axvline(
    1.0,
    linestyle="--",
)

plt.axvline(
    2.0,
    linestyle="--",
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Polynomial Interpolation and Extrapolation")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()