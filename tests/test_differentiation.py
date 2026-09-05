import unittest

import numpy as np

from differentiation.differentiation import (
    backward_derivative,
    backward_difference,
    central_difference,
    forward_derivative,
    forward_difference,
)


class TestFirstDerivativeMethods(unittest.TestCase):
    def test_forward_difference(self):
        def function(x):
            return x**2

        result = forward_difference(
            function,
            x=2.0,
            h=1.0e-5,
        )

        expected = 4.0

        self.assertAlmostEqual(result, expected, places=4)

    def test_backward_difference(self):
        def function(x):
            return x**2

        result = backward_difference(
            function,
            x=2.0,
            h=1.0e-5,
        )

        expected = 4.0

        self.assertAlmostEqual(result, expected, places=4)

    def test_central_difference(self):
        def function(x):
            return x**2

        result = central_difference(
            function,
            x=2.0,
            h=1.0e-5,
        )

        expected = 4.0

        self.assertAlmostEqual(result, expected, places=10)

    def test_forward_difference_invalid_step_size(self):
        def function(x):
            return x**2

        with self.assertRaises(ValueError):
            forward_difference(
                function,
                x=1.0,
                h=0.0,
            )

    def test_backward_difference_invalid_step_size(self):
        def function(x):
            return x**2

        with self.assertRaises(ValueError):
            backward_difference(
                function,
                x=1.0,
                h=-0.1,
            )

    def test_central_difference_invalid_step_size(self):
        def function(x):
            return x**2

        with self.assertRaises(ValueError):
            central_difference(
                function,
                x=1.0,
                h=0.0,
            )


class TestForwardDerivative(unittest.TestCase):
    def test_first_derivative_linear_data(self):
        x = np.linspace(0.0, 1.0, 11)
        h = x[1] - x[0]
        y = 3 * x + 2

        result = forward_derivative(
            y,
            h,
            derivative_order=1,
        )

        expected = np.full(result.size, 3.0)

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_second_derivative_quadratic_data(self):
        x = np.linspace(0.0, 1.0, 11)
        h = x[1] - x[0]
        y = x**2

        result = forward_derivative(
            y,
            h,
            derivative_order=2,
        )

        expected = np.full(result.size, 2.0)

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_invalid_step_size(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            forward_derivative(
                y,
                h=0.0,
                derivative_order=1,
            )

    def test_invalid_derivative_order_low(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            forward_derivative(
                y,
                h=1.0,
                derivative_order=0,
            )

    def test_invalid_derivative_order_high(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            forward_derivative(
                y,
                h=1.0,
                derivative_order=3,
            )


class TestBackwardDerivative(unittest.TestCase):
    def test_first_derivative_linear_data(self):
        x = np.linspace(0.0, 1.0, 11)
        h = x[1] - x[0]
        y = 3 * x + 2

        result = backward_derivative(
            y,
            h,
            derivative_order=1,
        )

        expected = np.full(result.size, 3.0)

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_second_derivative_quadratic_data(self):
        x = np.linspace(0.0, 1.0, 11)
        h = x[1] - x[0]
        y = x**2

        result = backward_derivative(
            y,
            h,
            derivative_order=2,
        )

        expected = np.full(result.size, 2.0)

        np.testing.assert_allclose(
            result,
            expected,
            atol=1.0e-12,
        )

    def test_invalid_step_size(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            backward_derivative(
                y,
                h=-1.0,
                derivative_order=1,
            )

    def test_invalid_derivative_order_low(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            backward_derivative(
                y,
                h=1.0,
                derivative_order=0,
            )

    def test_invalid_derivative_order_high(self):
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            backward_derivative(
                y,
                h=1.0,
                derivative_order=3,
            )


if __name__ == "__main__":
    unittest.main()