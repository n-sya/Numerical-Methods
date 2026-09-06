import json
import unittest
from pathlib import Path

import numpy as np

from differentiation.differentiation import central_difference
from integration.integration import simpson_rule
from interpolation.triangle_interpolation import barycentric_interpolation
from root_finding.root_finding import bisection


TEST_CASES_PATH = Path(__file__).with_name("test_cases.json")


class TestNumericalMethodsFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
            cls.test_cases = json.load(file)

    def test_simpson_rule(self):
        case = self.test_cases["integration"]

        x = np.array(case["x"], dtype=float)
        y = np.array(case["y"], dtype=float)

        result = simpson_rule(x, y)

        self.assertAlmostEqual(
            result,
            case["expected"],
            delta=case["tolerance"],
        )

    def test_central_difference(self):
        case = self.test_cases["differentiation"]

        function = lambda value: value**3

        result = central_difference(
            function,
            case["point"],
            case["step_size"],
        )

        self.assertAlmostEqual(
            result,
            case["expected"],
            delta=case["tolerance"],
        )

    def test_barycentric_interpolation(self):
        case = self.test_cases["triangle_interpolation"]

        vertices = np.array(case["vertices"], dtype=float)
        values = np.array(case["values"], dtype=float)
        point = np.array(case["point"], dtype=float)

        result = barycentric_interpolation(
            vertices,
            values,
            point,
        )

        self.assertAlmostEqual(
            result,
            case["expected"],
            delta=case["tolerance"],
        )

    def test_bisection(self):
        case = self.test_cases["root_finding"]

        function = lambda value: value**3 - value - 2.0

        root, _ = bisection(
            function,
            case["lower_bound"],
            case["upper_bound"],
        )

        self.assertAlmostEqual(
            root,
            case["expected"],
            delta=case["tolerance"],
        )


if __name__ == "__main__":
    unittest.main()