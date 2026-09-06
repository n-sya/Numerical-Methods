import numpy as np

from differentiation.differentiation import central_difference
from integration.integration import simpson_rule
from interpolation.triangle_interpolation import barycentric_interpolation
from root_finding.root_finding import bisection


def main():
    print("Numerical Methods for Engineering Analysis")
    print("=" * 42)

    # Numerical integration example
    x = np.linspace(0.0, 1.0, 101)
    y = x**2

    integral = simpson_rule(x, y)

    print("\nNumerical Integration")
    print(f"Integral of x^2 from 0 to 1: {integral:.6f}")

    # Numerical differentiation example
    function = lambda value: value**3
    derivative = central_difference(function, 2.0, 1e-5)

    print("\nNumerical Differentiation")
    print(f"Derivative of x^3 at x = 2: {derivative:.6f}")

    # Triangle interpolation example
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0])
    point = np.array([0.25, 0.25])

    interpolated_value = barycentric_interpolation(vertices, values, point)

    print("\nTriangle Interpolation")
    print(f"Interpolated value at {point}: {interpolated_value:.6f}")

    # Root-finding example
    function = lambda value: value**3 - value - 2.0
    root, _ = bisection(function, 1.0, 2.0)

    print("\nRoot Finding")
    print(f"Root of x^3 - x - 2: {root:.6f}")


if __name__ == "__main__":
    main()