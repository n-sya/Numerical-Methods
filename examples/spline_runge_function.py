import matplotlib.pyplot as plt
import numpy as np

from interpolation.interpolation import cubic_spline


# Runge function
def function(x):
    return 1 / (1 + 25 * x**2)


# Analytical derivative of the Runge function
def analytical_derivative(x):
    return (-50 * x) / (1 + 25 * x**2) ** 2


# Interpolation interval
a = -1.0
b = 1.0

# Interpolation nodes
number_of_nodes = 9

x_nodes = np.linspace(
    a,
    b,
    number_of_nodes,
)
y_nodes = function(x_nodes)

# Endpoint gradients
lower_gradient = analytical_derivative(a)
upper_gradient = analytical_derivative(b)

# Evaluation points
x_points = np.linspace(
    a,
    b,
    500,
)

# Calculate the cubic spline
y_spline = cubic_spline(
    x_nodes,
    y_nodes,
    lower_gradient,
    upper_gradient,
    x_points,
)

# Calculate the exact solution
y_exact = function(x_points)


# Plot the results
plt.figure(figsize=(8, 6))

plt.plot(
    x_points,
    y_exact,
    linewidth=2,
    label="Exact Function",
)

plt.plot(
    x_points,
    y_spline,
    label="Cubic Spline",
)

plt.scatter(
    x_nodes,
    y_nodes,
    label="Interpolation Nodes",
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Cubic Spline Interpolation of the Runge Function")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()