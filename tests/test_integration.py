import unittest

import numpy as np

from integration.integration import (
    adaptive_simpson,
    gauss_legendre_quadrature,
    simpson_rule,
    trapezoidal_rule,
)


class TestTrapezoidalRule(unittest.TestCase):
    def test_linear_function(self):
        x = np.linspace(0, 1, 11)
        y = 2 * x + 1

        result = trapezoidal_rule(x, y)
        expected = 2.0

        self.assertAlmostEqual(result, expected, places=12)

    def test_uneven_spacing(self):
        x = np.array([0.0, 0.1, 0.4, 1.0])
        y = 2 * x + 1

        result = trapezoidal_rule(x, y)
        expected = 2.0

        self.assertAlmostEqual(result, expected, places=12)

    def test_different_array_lengths(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            trapezoidal_rule(x, y)

    def test_insufficient_nodes(self):
        x = np.array([0.0])
        y = np.array([1.0])

        with self.assertRaises(ValueError):
            trapezoidal_rule(x, y)


class TestSimpsonRule(unittest.TestCase):
    def test_cubic_function(self):
        x = np.linspace(0, 1, 11)
        y = x**3

        result = simpson_rule(x, y)
        expected = 0.25

        self.assertAlmostEqual(result, expected, places=12)

    def test_different_array_lengths(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            simpson_rule(x, y)

    def test_insufficient_nodes(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])

        with self.assertRaises(ValueError):
            simpson_rule(x, y)

    def test_odd_number_of_subintervals(self):
        x = np.linspace(0, 1, 4)
        y = x**2

        with self.assertRaises(ValueError):
            simpson_rule(x, y)

    def test_uneven_spacing(self):
        x = np.array([0.0, 0.2, 0.5, 1.0, 1.5])
        y = x**2

        with self.assertRaises(ValueError):
            simpson_rule(x, y)


class TestAdaptiveSimpson(unittest.TestCase):
    def test_known_integral(self):
        def func(x):
            return 1 / (1 + x**2)

        result, estimated_error, number_of_subintervals = adaptive_simpson(
            func,
            0.0,
            1.0,
            tolerance=1.0e-10,
        )

        expected = np.pi / 4

        self.assertAlmostEqual(result, expected, places=9)
        self.assertLess(estimated_error, 1.0e-10)
        self.assertGreaterEqual(number_of_subintervals, 2)

    def test_invalid_limits(self):
        def func(x):
            return x**2

        with self.assertRaises(ValueError):
            adaptive_simpson(func, 1.0, 0.0)

    def test_invalid_tolerance(self):
        def func(x):
            return x**2

        with self.assertRaises(ValueError):
            adaptive_simpson(
                func,
                0.0,
                1.0,
                tolerance=0.0,
            )

    def test_invalid_max_iterations(self):
        def func(x):
            return x**2

        with self.assertRaises(ValueError):
            adaptive_simpson(
                func,
                0.0,
                1.0,
                max_iterations=0,
            )


class TestGaussLegendreQuadrature(unittest.TestCase):
    def test_quadratic_function(self):
        def func(x):
            return x**2

        result = gauss_legendre_quadrature(
            func,
            0.0,
            1.0,
            number_of_nodes=2,
        )

        expected = 1 / 3

        self.assertAlmostEqual(result, expected, places=12)

    def test_known_integral(self):
        def func(x):
            return 1 / (1 + x**2)

        result = gauss_legendre_quadrature(
            func,
            0.0,
            1.0,
            number_of_nodes=5,
        )

        expected = np.pi / 4

        self.assertAlmostEqual(result, expected, places=6)

    def test_invalid_number_of_nodes_low(self):
        def func(x):
            return x

        with self.assertRaises(ValueError):
            gauss_legendre_quadrature(
                func,
                0.0,
                1.0,
                number_of_nodes=0,
            )

    def test_invalid_number_of_nodes_high(self):
        def func(x):
            return x

        with self.assertRaises(ValueError):
            gauss_legendre_quadrature(
                func,
                0.0,
                1.0,
                number_of_nodes=6,
            )

    def test_invalid_limits(self):
        def func(x):
            return x

        with self.assertRaises(ValueError):
            gauss_legendre_quadrature(
                func,
                1.0,
                0.0,
                number_of_nodes=2,
            )


if __name__ == "__main__":
    unittest.main()