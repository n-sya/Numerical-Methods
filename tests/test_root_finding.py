import unittest

import numpy as np

from root_finding.root_finding import (
    bisection,
    newton_raphson,
    newton_system,
    numerical_jacobian,
)


class TestBisection(unittest.TestCase):
    def test_quadratic_root(self):
        def function(x):
            return x**2 - 2.0

        root, iterations = bisection(
            function,
            a=0.0,
            b=2.0,
        )

        self.assertAlmostEqual(
            root,
            np.sqrt(2.0),
            places=7,
        )

        self.assertGreater(
            iterations,
            0,
        )

    def test_invalid_bracket(self):
        def function(x):
            return x**2 + 1.0

        with self.assertRaises(ValueError):
            bisection(
                function,
                a=-1.0,
                b=1.0,
            )

    def test_invalid_tolerance(self):
        def function(x):
            return x

        with self.assertRaises(ValueError):
            bisection(
                function,
                a=-1.0,
                b=1.0,
                tolerance=0.0,
            )


class TestNewtonRaphson(unittest.TestCase):
    def test_quadratic_root(self):
        def function(x):
            return x**2 - 2.0

        def derivative(x):
            return 2.0 * x

        root, iterations = newton_raphson(
            function,
            derivative,
            initial_guess=1.0,
        )

        self.assertAlmostEqual(
            root,
            np.sqrt(2.0),
            places=7,
        )

        self.assertGreater(
            iterations,
            0,
        )

    def test_zero_derivative(self):
        def function(x):
            return x**2 + 1.0

        def derivative(x):
            return 2.0 * x

        with self.assertRaises(RuntimeError):
            newton_raphson(
                function,
                derivative,
                initial_guess=0.0,
            )


class TestNumericalJacobian(unittest.TestCase):
    def test_known_jacobian(self):
        def function(x):
            return np.array(
                [
                    x[0] ** 2 + x[1],
                    x[0] + x[1] ** 2,
                ]
            )

        point = np.array(
            [
                1.0,
                2.0,
            ]
        )

        jacobian = numerical_jacobian(
            function,
            point,
        )

        expected = np.array(
            [
                [2.0, 1.0],
                [1.0, 4.0],
            ]
        )

        np.testing.assert_allclose(
            jacobian,
            expected,
            atol=1.0e-5,
        )


class TestNewtonSystem(unittest.TestCase):
    def test_two_equation_system(self):
        # x + y = 3
        # x - y = 1
        # solution: x = 2, y = 1

        def function(x):
            return np.array(
                [
                    x[0] + x[1] - 3.0,
                    x[0] - x[1] - 1.0,
                ]
            )

        solution, iterations = newton_system(
            function,
            initial_guess=np.array(
                [
                    0.0,
                    0.0,
                ]
            ),
        )

        expected = np.array(
            [
                2.0,
                1.0,
            ]
        )

        np.testing.assert_allclose(
            solution,
            expected,
            atol=1.0e-8,
        )

        self.assertGreater(
            iterations,
            0,
        )


if __name__ == "__main__":
    unittest.main()