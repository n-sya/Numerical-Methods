import matplotlib.pyplot as plt
import numpy as np

from ode.boundary_value import shooting_method


# Blasius equation written as a first-order system:
# f' = g
# g' = h
# h' = -0.5*f*h
def blasius_system(eta, y):
    derivative = np.zeros(3)

    derivative[0] = y[1]
    derivative[1] = y[2]
    derivative[2] = -0.5 * y[0] * y[2]

    return derivative


# Construct the initial conditions from a guess of f''(0)
def initial_state_function(initial_second_derivative):
    return np.array(
        [
            0.0,
            0.0,
            initial_second_derivative,
        ]
    )


# Boundary condition residual at eta = 10
def residual_function(eta, solution):
    return solution[1, -1] - 1.0


# Solve the boundary-value problem
eta, solution, initial_second_derivative, iterations = (
    shooting_method(
        blasius_system,
        initial_state_function,
        residual_function,
        t0=0.0,
        t_end=10.0,
        h=0.01,
        first_guess=0.2,
        second_guess=0.5,
    )
)

f = solution[0]
velocity = solution[1]


# Estimate the 99% boundary-layer location
difference = np.abs(velocity - 0.99)
eta_99_index = np.argmin(difference)
eta_99 = eta[eta_99_index]


# Calculate displacement and momentum thickness
displacement_thickness = np.trapz(
    1.0 - velocity,
    eta,
)

momentum_thickness = np.trapz(
    velocity * (1.0 - velocity),
    eta,
)


# Print results
print(
    "Estimated f''(0):",
    initial_second_derivative,
)

print(
    "Iterations:",
    iterations,
)

print(
    "Eta at 99% velocity:",
    eta_99,
)

print(
    "Displacement thickness:",
    displacement_thickness,
)

print(
    "Momentum thickness:",
    momentum_thickness,
)


# Plot the velocity profile
plt.figure(figsize=(8, 6))

plt.plot(
    velocity,
    eta,
    linewidth=2,
    label="Velocity Profile",
)

plt.axhline(
    eta_99,
    linestyle="--",
    label="99% Boundary Layer",
)

plt.xlabel("Normalised Velocity")
plt.ylabel("Similarity Coordinate, eta")
plt.title("Blasius Boundary Layer Solution")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/blasius_boundary_layer.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()