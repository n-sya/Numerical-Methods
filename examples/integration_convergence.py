import matplotlib.pyplot as plt
import numpy as np

from integration.integration import simpson_rule, trapezoidal_rule


# Function with a known analytical integral
def function(x):
    return 1 / (1 + x**2)


# Analytical solution over the interval [0, 1]
analytical_integral = np.pi / 4

# Use even numbers of subintervals so Simpson's rule is always valid
subintervals = np.array([2, 4, 8, 16, 32, 64, 128, 256])

trapezoidal_errors = []
simpson_errors = []


# Calculate numerical error for progressively refined grids
for number_of_subintervals in subintervals:
    x = np.linspace(
        0.0,
        1.0,
        number_of_subintervals + 1,
    )
    y = function(x)

    trapezoidal_result = trapezoidal_rule(x, y)
    simpson_result = simpson_rule(x, y)

    trapezoidal_error = abs(
        trapezoidal_result - analytical_integral
    )
    simpson_error = abs(
        simpson_result - analytical_integral
    )

    trapezoidal_errors.append(trapezoidal_error)
    simpson_errors.append(simpson_error)


# Convert results to NumPy arrays
trapezoidal_errors = np.array(trapezoidal_errors)
simpson_errors = np.array(simpson_errors)


# Plot convergence behaviour
plt.figure(figsize=(8, 6))

plt.loglog(
    subintervals,
    trapezoidal_errors,
    marker="o",
    label="Trapezoidal Rule",
)

plt.loglog(
    subintervals,
    simpson_errors,
    marker="s",
    label="Simpson's Rule",
)

plt.xlabel("Number of Subintervals")
plt.ylabel("Absolute Error")
plt.title("Numerical Integration Convergence")
plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.show()