import matplotlib.pyplot as plt
import numpy as np

from root_finding.root_finding import (
    bisection,
    newton_raphson,
)


# Nonlinear equation
def function(x):
    return x**3 - x - 2.0


# Analytical derivative
def derivative(x):
    return 3.0 * x**2 - 1.0


# Solve using the bisection method
bisection_root, bisection_iterations = bisection(
    function,
    a=1.0,
    b=2.0,
)


# Solve using Newton-Raphson
newton_root, newton_iterations = newton_raphson(
    function,
    derivative,
    initial_guess=1.5,
)


# Print comparison
print("Bisection method")
print(
    f"Root: {bisection_root:.10f}"
)
print(
    f"Iterations: {bisection_iterations}"
)

print()

print("Newton-Raphson method")
print(
    f"Root: {newton_root:.10f}"
)
print(
    f"Iterations: {newton_iterations}"
)


# Plot the nonlinear function and computed root
x = np.linspace(
    0.5,
    2.5,
    400,
)

y = function(x)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    x,
    y,
    label="f(x)",
)

plt.axhline(
    0.0,
    linewidth=1,
)

plt.scatter(
    bisection_root,
    0.0,
    label="Computed Root",
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(
    "Nonlinear Root Finding"
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()