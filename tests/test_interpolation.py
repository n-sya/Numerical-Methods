import unittest

import numpy as np

from interpolation.interpolation import (
    cubic_spline,
    gaussian_elimination,
    lagrange_interpolation,
    newton_divided_difference_iterative,
    newton_divided_difference_recursive,
    newton_interpolation,
)


class TestGaussianElimination(unittest.TestCase):
    def test_linear_system(self):
        A = np.array(
            [
                [2.0, 1.0],
                [1.0, 3.0],
            ]
        )
        b = np.array([5.0, 6.0])

        result = gaussian_elimination(A, b)
        expected = np.array([1.8, 1.4])

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_non_square_matrix(self):
        A = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        b = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            gaussian_elimination(A, b)


class TestLagrangeInterpolation(unittest.TestCase):
    def test_quadratic_polynomial(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = x_nodes**2

        x_points = np.array([0.5, 1.5])

        result = lagrange_interpolation(
            x_nodes,
            y_nodes,
            x_points,
        )

        expected = x_points**2

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_interpolation_at_nodes(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = np.array([1.0, 3.0, 2.0])

        result = lagrange_interpolation(
            x_nodes,
            y_nodes,
            x_nodes,
        )

        np.testing.assert_allclose(
            result,
            y_nodes,
            atol=1.0e-12,
        )

    def test_different_array_lengths(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            lagrange_interpolation(
                x_nodes,
                y_nodes,
                np.array([0.5]),
            )


class TestNewtonDividedDifference(unittest.TestCase):
    def test_recursive_quadratic(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = x_nodes**2

        result = newton_divided_difference_recursive(
            x_nodes,
            y_nodes,
        )

        expected = 1.0

        self.assertAlmostEqual(
            result,
            expected,
            places=12,
        )

    def test_iterative_quadratic(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = x_nodes**2

        result = newton_divided_difference_iterative(
            x_nodes,
            y_nodes,
        )

        expected = 1.0

        self.assertAlmostEqual(
            result,
            expected,
            places=12,
        )

    def test_recursive_and_iterative_agree(self):
        x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
        y_nodes = np.sin(x_nodes)

        recursive_result = newton_divided_difference_recursive(
            x_nodes,
            y_nodes,
        )

        iterative_result = newton_divided_difference_iterative(
            x_nodes,
            y_nodes,
        )

        self.assertAlmostEqual(
            recursive_result,
            iterative_result,
            places=12,
        )


class TestNewtonInterpolation(unittest.TestCase):
    def test_quadratic_polynomial(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = x_nodes**2

        x_points = np.array([0.5, 1.5])

        result = newton_interpolation(
            x_nodes,
            y_nodes,
            x_points,
        )

        expected = x_points**2

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_lagrange_and_newton_agree(self):
        x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
        y_nodes = np.sin(x_nodes)

        x_points = np.linspace(0.0, 3.0, 20)

        lagrange_result = lagrange_interpolation(
            x_nodes,
            y_nodes,
            x_points,
        )

        newton_result = newton_interpolation(
            x_nodes,
            y_nodes,
            x_points,
        )

        np.testing.assert_allclose(
            newton_result,
            lagrange_result,
            atol=1.0e-12,
        )


class TestCubicSpline(unittest.TestCase):
    def test_interpolation_at_nodes(self):
        x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
        y_nodes = x_nodes**3

        lower_gradient = 0.0
        upper_gradient = 27.0

        result = cubic_spline(
            x_nodes,
            y_nodes,
            lower_gradient,
            upper_gradient,
            x_nodes,
        )

        np.testing.assert_allclose(
            result,
            y_nodes,
            atol=1.0e-12,
        )

    def test_cubic_function(self):
        x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
        y_nodes = x_nodes**3

        x_points = np.linspace(0.0, 3.0, 31)

        result = cubic_spline(
            x_nodes,
            y_nodes,
            lower_gradient=0.0,
            upper_gradient=27.0,
            x_points=x_points,
        )

        expected = x_points**3

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-10,
        )

    def test_outside_interpolation_range(self):
        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = x_nodes**2

        x_points = np.array([-1.0, 0.5, 3.0])

        result = cubic_spline(
            x_nodes,
            y_nodes,
            lower_gradient=0.0,
            upper_gradient=4.0,
            x_points=x_points,
        )

        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[2]))

    def test_non_increasing_nodes(self):
        x_nodes = np.array([0.0, 2.0, 1.0])
        y_nodes = np.array([0.0, 4.0, 1.0])

        with self.assertRaises(ValueError):
            cubic_spline(
                x_nodes,
                y_nodes,
                lower_gradient=0.0,
                upper_gradient=2.0,
                x_points=np.array([0.5]),
            )


def test_partial_pivoting(self):
    A = np.array(
        [
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    b = np.array(
        [
            4.0,
            3.0,
        ]
    )

    solution = gaussian_elimination(
        A,
        b,
    )

    expected = np.array(
        [
            1.0,
            2.0,
        ]
    )

    np.testing.assert_allclose(
        solution,
        expected,
        atol=1.0e-10,
    )

if __name__ == "__main__":
    unittest.main()