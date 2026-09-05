import unittest

import numpy as np

from ode.initial_value import (
    backward_euler,
    forward_euler,
    forward_euler_system,
    runge_kutta_4,
    runge_kutta_4_system,
)


class TestForwardEuler(unittest.TestCase):
    def test_exponential_decay(self):
        def function(t, y):
            return -y

        t, y = forward_euler(
            function,
            t0=0.0,
            y0=1.0,
            h=0.001,
            t_end=1.0,
        )

        expected = np.exp(-1.0)

        self.assertAlmostEqual(
            y[-1],
            expected,
            places=3,
        )

    def test_initial_condition(self):
        def function(t, y):
            return -y

        t, y = forward_euler(
            function,
            t0=0.0,
            y0=2.0,
            h=0.1,
            t_end=1.0,
        )

        self.assertEqual(t[0], 0.0)
        self.assertEqual(y[0], 2.0)

    def test_invalid_step_size(self):
        def function(t, y):
            return -y

        with self.assertRaises(ValueError):
            forward_euler(
                function,
                t0=0.0,
                y0=1.0,
                h=0.0,
                t_end=1.0,
            )


class TestRungeKutta4(unittest.TestCase):
    def test_exponential_decay(self):
        def function(t, y):
            return -y

        t, y = runge_kutta_4(
            function,
            t0=0.0,
            y0=1.0,
            h=0.1,
            t_end=1.0,
        )

        expected = np.exp(-1.0)

        self.assertAlmostEqual(
            y[-1],
            expected,
            places=6,
        )

    def test_rk4_more_accurate_than_forward_euler(self):
        def function(t, y):
            return -y

        _, forward_solution = forward_euler(
            function,
            t0=0.0,
            y0=1.0,
            h=0.1,
            t_end=1.0,
        )

        _, rk4_solution = runge_kutta_4(
            function,
            t0=0.0,
            y0=1.0,
            h=0.1,
            t_end=1.0,
        )

        expected = np.exp(-1.0)

        forward_error = abs(
            forward_solution[-1] - expected
        )

        rk4_error = abs(
            rk4_solution[-1] - expected
        )

        self.assertLess(
            rk4_error,
            forward_error,
        )


class TestBackwardEuler(unittest.TestCase):
    def test_exponential_decay(self):
        def function(t, y):
            return -y

        def derivative_y(t, y):
            return -1.0

        t, y = backward_euler(
            function,
            derivative_y,
            t0=0.0,
            y0=1.0,
            h=0.001,
            t_end=1.0,
        )

        expected = np.exp(-1.0)

        self.assertAlmostEqual(
            y[-1],
            expected,
            places=3,
        )

    def test_invalid_tolerance(self):
        def function(t, y):
            return -y

        def derivative_y(t, y):
            return -1.0

        with self.assertRaises(ValueError):
            backward_euler(
                function,
                derivative_y,
                t0=0.0,
                y0=1.0,
                h=0.1,
                t_end=1.0,
                tolerance=0.0,
            )


class TestForwardEulerSystem(unittest.TestCase):
    def test_two_equation_system(self):
        def function(t, y):
            return np.array(
                [
                    y[1],
                    -y[0],
                ]
            )

        y0 = np.array([1.0, 0.0])

        t, y = forward_euler_system(
            function,
            y0,
            t0=0.0,
            t_end=1.0,
            h=0.001,
        )

        expected_position = np.cos(1.0)
        expected_velocity = -np.sin(1.0)

        self.assertAlmostEqual(
            y[0, -1],
            expected_position,
            places=2,
        )

        self.assertAlmostEqual(
            y[1, -1],
            expected_velocity,
            places=2,
        )

    def test_wrong_derivative_size(self):
        def function(t, y):
            return np.array([y[0]])

        y0 = np.array([1.0, 0.0])

        with self.assertRaises(ValueError):
            forward_euler_system(
                function,
                y0,
                t0=0.0,
                t_end=1.0,
                h=0.1,
            )


class TestRungeKutta4System(unittest.TestCase):
    def test_two_equation_system(self):
        def function(t, y):
            return np.array(
                [
                    y[1],
                    -y[0],
                ]
            )

        y0 = np.array([1.0, 0.0])

        t, y = runge_kutta_4_system(
            function,
            y0,
            t0=0.0,
            t_end=1.0,
            h=0.1,
        )

        expected_position = np.cos(1.0)
        expected_velocity = -np.sin(1.0)

        self.assertAlmostEqual(
            y[0, -1],
            expected_position,
            places=5,
        )

        self.assertAlmostEqual(
            y[1, -1],
            expected_velocity,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()