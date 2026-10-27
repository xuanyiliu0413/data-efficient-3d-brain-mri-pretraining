import unittest
from analysis.paired_bootstrap import paired_bootstrap

class TestStatistics(unittest.TestCase):
    def test_constant_difference(self):
        result = paired_bootstrap(
            [0.5, 0.6, 0.7],
            [0.6, 0.7, 0.8],
            iterations=100,
            seed=42
        )
        self.assertAlmostEqual(result["estimate"], 0.1)
        self.assertAlmostEqual(result["ci_lower"], 0.1)
        self.assertAlmostEqual(result["ci_upper"], 0.1)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([0.5], [])
