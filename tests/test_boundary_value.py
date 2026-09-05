import unittest

import numpy as np

from ode.boundary_value import (
    finite_difference_dirichlet,
    finite_difference_robin,
    shooting_method,
)


class TestDirichletBoundaryValue(unittest.TestCase):
    def test_linear_solution(self):
        # y'' = 0
        # y(0) = 1
        # y(1) = 3
        # analytical solution: y = 1 + 2x

        def coefficients(x):
            return 0.0, 0.0, 0.0

        x, y = finite_difference_dirichlet(
            coefficients,
            a=0.0,
            b=1.0,
            y_a=1.0,
            y_b=3.0,
            number_of_intervals=20,
        )

        expected = 1.0 + 2.0 * x

        self.assertTrue(
            np.allclose(
                y,
                expected,
                atol=1.0e-10,
            )
        )

    def test_quadratic_solution(self):
        # y'' = 2
        # y(0) = 0
        # y(1) = 1
        # analytical solution: y = x^2

        def coefficients(x):
            return 0.0, 0.0, 2.0

        x, y = finite_difference_dirichlet(
            coefficients,
            a=0.0,
            b=1.0,
            y_a=0.0,
            y_b=1.0,
            number_of_intervals=20,
        )

        expected = x**2

        self.assertTrue(
            np.allclose(
                y,
                expected,
                atol=1.0e-10,
            )
        )

    def test_invalid_domain(self):
        def coefficients(x):
            return 0.0, 0.0, 0.0

        with self.assertRaises(ValueError):
            finite_difference_dirichlet(
                coefficients,
                a=1.0,
                b=0.0,
                y_a=0.0,
                y_b=1.0,
                number_of_intervals=20,
            )

    def test_invalid_number_of_intervals(self):
        def coefficients(x):
            return 0.0, 0.0, 0.0

        with self.assertRaises(ValueError):
            finite_difference_dirichlet(
                coefficients,
                a=0.0,
                b=1.0,
                y_a=0.0,
                y_b=1.0,
                number_of_intervals=1,
            )


class TestRobinBoundaryValue(unittest.TestCase):
    def test_derivative_and_value_boundaries(self):
        # y'' = 0
        # y'(0) = 2
        # y(1) = 3
        # analytical solution: y = 1 + 2x

        def coefficients(x):
            return 0.0, 0.0, 0.0

        left_boundary = (
            1.0,
            0.0,
            2.0,
        )

        right_boundary = (
            0.0,
            1.0,
            3.0,
        )

        x, y = finite_difference_robin(
            coefficients,
            a=0.0,
            b=1.0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            number_of_intervals=20,
        )

        expected = 1.0 + 2.0 * x

        self.assertTrue(
            np.allclose(
                y,
                expected,
                atol=1.0e-10,
            )
        )

    def test_value_and_derivative_boundaries(self):
        # y'' = 0
        # y(0) = 1
        # y'(1) = 2
        # analytical solution: y = 1 + 2x

        def coefficients(x):
            return 0.0, 0.0, 0.0

        left_boundary = (
            0.0,
            1.0,
            1.0,
        )

        right_boundary = (
            1.0,
            0.0,
            2.0,
        )

        x, y = finite_difference_robin(
            coefficients,
            a=0.0,
            b=1.0,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            number_of_intervals=20,
        )

        expected = 1.0 + 2.0 * x

        self.assertTrue(
            np.allclose(
                y,
                expected,
                atol=1.0e-10,
            )
        )

    def test_invalid_domain(self):
        def coefficients(x):
            return 0.0, 0.0, 0.0

        with self.assertRaises(ValueError):
            finite_difference_robin(
                coefficients,
                a=1.0,
                b=0.0,
                left_boundary=(1.0, 0.0, 0.0),
                right_boundary=(0.0, 1.0, 0.0),
                number_of_intervals=20,
            )


class TestShootingMethod(unittest.TestCase):
    def test_linear_boundary_value_problem(self):
        # y'' = 0
        # y(0) = 1
        # y(1) = 3
        # exact solution: y = 1 + 2x

        def system(x, y):
            return np.array(
                [
                    y[1],
                    0.0,
                ]
            )

        def initial_state_function(initial_slope):
            return np.array(
                [
                    1.0,
                    initial_slope,
                ]
            )

        def residual_function(x, solution):
            return solution[0, -1] - 3.0

        x, solution, initial_slope, iterations = (
            shooting_method(
                system,
                initial_state_function,
                residual_function,
                t0=0.0,
                t_end=1.0,
                h=0.01,
                first_guess=1.0,
                second_guess=3.0,
            )
        )

        self.assertAlmostEqual(
            initial_slope,
            2.0,
            places=7,
        )

        np.testing.assert_allclose(
            solution[0],
            1.0 + 2.0 * x,
            atol=1.0e-8,
        )

        self.assertGreater(
            iterations,
            0,
        )

if __name__ == "__main__":
    unittest.main()