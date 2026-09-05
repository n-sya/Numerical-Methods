import matplotlib.pyplot as plt
import numpy as np

from ode.initial_value import runge_kutta_4_system


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


# Solve the Blasius problem for a given guess of f''(0)
def solve_with_guess(initial_second_derivative):
    initial_conditions = np.array(
        [
            0.0,
            0.0,
            initial_second_derivative,
        ]
    )

    eta, solution = runge_kutta_4_system(
        blasius_system,
        initial_conditions,
        t0=0.0,
        t_end=10.0,
        h=0.01,
    )

    return eta, solution


# Shooting method using the secant method
def shooting_method(
    first_guess,
    second_guess,
    tolerance=1.0e-8,
    max_iterations=50,
):
    guess_previous = first_guess
    guess_current = second_guess

    _, solution_previous = solve_with_guess(
        guess_previous
    )

    error_previous = (
        solution_previous[1, -1] - 1.0
    )

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        eta, solution_current = solve_with_guess(
            guess_current
        )

        error_current = (
            solution_current[1, -1] - 1.0
        )

        if abs(error_current) < tolerance:
            return (
                eta,
                solution_current,
                guess_current,
                iteration,
            )

        denominator = (
            error_current - error_previous
        )

        if np.isclose(
            denominator,
            0.0,
        ):
            raise RuntimeError(
                "The shooting method encountered "
                "a zero secant denominator."
            )

        guess_next = (
            guess_current
            - error_current
            * (
                guess_current
                - guess_previous
            )
            / denominator
        )

        guess_previous = guess_current
        error_previous = error_current
        guess_current = guess_next

    raise RuntimeError(
        "The shooting method did not converge within "
        "the maximum number of iterations."
    )


# Solve the boundary-value problem
eta, solution, initial_second_derivative, iterations = (
    shooting_method(
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
plt.show()