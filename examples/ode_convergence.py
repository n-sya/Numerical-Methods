import matplotlib.pyplot as plt
import numpy as np

from ode.initial_value import (
    backward_euler,
    forward_euler,
    runge_kutta_4,
)


# Test problem with analytical solution
def function(t, y):
    return -y


# Derivative of the ODE function with respect to y
def derivative_y(t, y):
    return -1.0


# Analytical solution
def analytical_solution(t):
    return np.exp(-t)


# Problem definition
t0 = 0.0
t_end = 1.0
y0 = 1.0

# Step sizes to test
step_sizes = np.array(
    [
        0.2,
        0.1,
        0.05,
        0.025,
        0.0125,
    ]
)

forward_errors = []
backward_errors = []
rk4_errors = []


# Calculate error at the final time
for h in step_sizes:
    _, forward_solution = forward_euler(
        function,
        t0,
        y0,
        h,
        t_end,
    )

    _, backward_solution = backward_euler(
        function,
        derivative_y,
        t0,
        y0,
        h,
        t_end,
    )

    _, rk4_solution = runge_kutta_4(
        function,
        t0,
        y0,
        h,
        t_end,
    )

    exact = analytical_solution(t_end)

    forward_errors.append(
        abs(forward_solution[-1] - exact)
    )

    backward_errors.append(
        abs(backward_solution[-1] - exact)
    )

    rk4_errors.append(
        abs(rk4_solution[-1] - exact)
    )


# Convert results to NumPy arrays
forward_errors = np.array(forward_errors)
backward_errors = np.array(backward_errors)
rk4_errors = np.array(rk4_errors)


# Plot convergence
plt.figure(figsize=(8, 6))

plt.loglog(
    step_sizes,
    forward_errors,
    marker="o",
    label="Forward Euler",
)

plt.loglog(
    step_sizes,
    backward_errors,
    marker="s",
    label="Backward Euler",
)

plt.loglog(
    step_sizes,
    rk4_errors,
    marker="^",
    label="RK4",
)

plt.xlabel("Step Size, h")
plt.ylabel("Absolute Error at t = 1")
plt.title("ODE Solver Convergence")
plt.grid(True, which="both")
plt.legend()

plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    "outputs/ode_convergence.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()