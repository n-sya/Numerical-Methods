import matplotlib.pyplot as plt
import numpy as np

from differentiation.differentiation import (
    backward_difference,
    central_difference,
    forward_difference,
)


# Function and analytical derivative
def function(x):
    return np.sin(x)


def analytical_derivative(x):
    return np.cos(x)


# Point at which the derivative is evaluated
x = 1.0

# Range of step sizes
step_sizes = np.logspace(-1, -6, 20)

forward_errors = []
backward_errors = []
central_errors = []


# Calculate the error for each finite difference method
for h in step_sizes:
    exact = analytical_derivative(x)

    forward_result = forward_difference(
        function,
        x,
        h,
    )

    backward_result = backward_difference(
        function,
        x,
        h,
    )

    central_result = central_difference(
        function,
        x,
        h,
    )

    forward_errors.append(
        abs(forward_result - exact)
    )

    backward_errors.append(
        abs(backward_result - exact)
    )

    central_errors.append(
        abs(central_result - exact)
    )


# Convert results to NumPy arrays
forward_errors = np.array(forward_errors)
backward_errors = np.array(backward_errors)
central_errors = np.array(central_errors)


# Plot convergence behaviour
plt.figure(figsize=(8, 6))

plt.loglog(
    step_sizes,
    forward_errors,
    marker="o",
    label="Forward Difference",
)

plt.loglog(
    step_sizes,
    backward_errors,
    marker="s",
    label="Backward Difference",
)

plt.loglog(
    step_sizes,
    central_errors,
    marker="^",
    label="Central Difference",
)

plt.xlabel("Step Size, h")
plt.ylabel("Absolute Error")
plt.title("Finite Difference Convergence")
plt.grid(True, which="both")
plt.legend()

plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    "outputs/differentiation_convergence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()