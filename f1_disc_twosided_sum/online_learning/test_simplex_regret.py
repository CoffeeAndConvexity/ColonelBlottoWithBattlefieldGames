import unittest
import numpy as np
from online_learning.regret_matching import RegretMatching
from online_learning.dag_structure import DagStructure
from online_learning.dag_regret_minimizer import DagRegretMinimizer

class TestRegretMatching(unittest.TestCase):
    def test_recommend_uniform_distribution(self):
        """Test if recommend returns a uniform distribution when regrets are zero."""
        n = 3
        rm = RegretMatching(n)
        strategy = rm.recommend()
        expected = np.ones(n) / n
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_update_regrets(self):
        """Test if update_regrets updates the regrets correctly."""
        n = 3
        rm = RegretMatching(n)
        rewards = np.array([1, 0, 0])
        last_strategy = np.array([0.5, 0.25, 0.25])
        rm.update_regrets(rewards, last_strategy)
        expected_regrets = np.array([0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(rm.regrets, expected_regrets)

    def test_recommend_based_on_regrets(self):
        """Test if recommend generates probabilities based on positive regrets."""
        n = 3
        rm = RegretMatching(n)
        rm.regrets = np.array([1, 0, -1])
        strategy = rm.recommend()
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_solve_matrix_game(self):
        """Test if solve_matrix_game produces valid average strategies."""
        matrix = np.array([[1, 0], [0, 2]])
        avg_strategy_p1, avg_strategy_p2 = RegretMatching.solve_matrix_game(matrix, iterations=10000)

        # Check if the strategies are valid probability distributions
        self.assertAlmostEqual(np.sum(avg_strategy_p1), 1.0)
        self.assertAlmostEqual(np.sum(avg_strategy_p2), 1.0)
        self.assertTrue(np.all(avg_strategy_p1 >= 0))
        self.assertTrue(np.all(avg_strategy_p2 >= 0))

        # Check that expected strategies are close to the Nash equilibrium
        ne_p1 = np.array([2./3, 1./3])
        ne_p2 = np.array([2./3, 1./3])
        np.testing.assert_array_almost_equal(avg_strategy_p1, ne_p1, decimal=2)
        np.testing.assert_array_almost_equal(avg_strategy_p2, ne_p2, decimal=2)

if __name__ == '__main__':
    unittest.main()