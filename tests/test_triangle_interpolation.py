import unittest

import numpy as np

from interpolation.triangle_interpolation import (
    barycentric_coordinates,
    barycentric_interpolation,
    inverse_distance_interpolation,
)


class TestBarycentricCoordinates(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def test_coordinates_sum_to_one(self):
        point = np.array([0.25, 0.25])

        coordinates = barycentric_coordinates(self.vertices, point)

        self.assertAlmostEqual(np.sum(coordinates), 1.0)

    def test_coordinates_at_vertex(self):
        point = self.vertices[1]

        coordinates = barycentric_coordinates(self.vertices, point)

        expected = np.array([0.0, 1.0, 0.0])

        np.testing.assert_allclose(coordinates, expected)

    def test_collinear_vertices_raise_error(self):
        vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        point = np.array([0.5, 0.5])

        with self.assertRaises(ValueError):
            barycentric_coordinates(vertices, point)


class TestBarycentricInterpolation(unittest.TestCase):
    def test_reproduces_linear_function(self):
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        point = np.array([0.25, 0.5])

        values = 2.0 * vertices[:, 0] + 3.0 * vertices[:, 1] + 1.0
        expected = 2.0 * point[0] + 3.0 * point[1] + 1.0

        interpolated = barycentric_interpolation(vertices, values, point)

        self.assertAlmostEqual(interpolated, expected)


class TestInverseDistanceInterpolation(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.values = np.array([1.0, 2.0, 3.0])

    def test_constant_values_are_preserved(self):
        values = np.array([5.0, 5.0, 5.0])
        point = np.array([0.25, 0.25])

        interpolated = inverse_distance_interpolation(
            self.vertices,
            values,
            point,
        )

        self.assertAlmostEqual(interpolated, 5.0)

    def test_point_at_vertex_returns_vertex_value(self):
        point = self.vertices[2]

        interpolated = inverse_distance_interpolation(
            self.vertices,
            self.values,
            point,
        )

        self.assertAlmostEqual(interpolated, self.values[2])


if __name__ == "__main__":
    unittest.main()