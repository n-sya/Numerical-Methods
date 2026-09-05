import matplotlib.pyplot as plt
import numpy as np

from interpolation.interpolation import newton_interpolation


# Runge function
def function(x):
    return 1 / (1 + 25 * x**2)


# Interpolation interval
a = -1.0
b = 1.0

# Evaluation points for plotting
x_points = np.linspace(a, b, 500)
y_exact = function(x_points)

# Polynomial degrees to compare
polynomial_degrees = [2, 4, 8, 12]


# Plot the analytical function
plt.figure(figsize=(8, 6))

plt.plot(
    x_points,
    y_exact,
    linewidth=2,
    label="Exact Function",
)


# Construct Newton interpolating polynomials
for degree in polynomial_degrees:
    number_of_nodes = degree + 1

    x_nodes = np.linspace(
        a,
        b,
        number_of_nodes,
    )
    y_nodes = function(x_nodes)

    y_interpolated = newton_interpolation(
        x_nodes,
        y_nodes,
        x_points,
    )

    plt.plot(
        x_points,
        y_interpolated,
        label=f"Degree {degree}",
    )


plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Runge's Phenomenon in Polynomial Interpolation")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()